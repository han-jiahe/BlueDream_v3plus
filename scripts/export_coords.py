#!/usr/bin/env python3
"""
BlueDream v3plus — Video to Pitch Coordinates Exporter

Full Soccer_Analysis pipeline:
  1. YOLOv11n soccana → Player/Ball/Referee detection
  2. ByteTrack → Multi-object tracking with consistent IDs
  3. SigLIP + UMAP + KMeans → Team assignment
  4. YOLOv11n-pose → 29 pitch keypoints → Homography → real-world coords

Output CSV (v3-compatible):
  帧号, 球员ID, 画面X, 画面Y, 雷达X, 雷达Y, 真实X(米), 真实Y(米), 队伍

Video output modes (--viz):
  none      CSV only (fastest)
  bbox      Player bounding boxes + IDs + team colors
  radar     Tactical radar mini-map overlay on original frame
  keypoints Pitch keypoints + connections overlay
  all       Combined: bbox + radar + keypoints

Usage:
    # CSV only (fastest)
    python scripts/export_coords.py --source video.mp4 --device cuda

    # CSV + annotated video with radar overlay
    python scripts/export_coords.py --source video.mp4 --device cuda --viz radar --video out.mp4

    # Show all visualizations
    python scripts/export_coords.py --source video.mp4 --device cuda --viz all --video out.mp4
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Path setup ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "soccana"))

from constants import model_path as _default_model_path
from keypoint_detection.keypoint_constants import (
    keypoint_model_path as _default_keypoint_path,
    KEYPOINT_CONNECTIONS,
    FIFA_FIELD_LENGTH,
    FIFA_FIELD_WIDTH,
)

# Pipeline components
from player_detection import load_detection_model, get_detections
from keypoint_detection import load_keypoint_model, get_keypoint_detections
from player_tracking import TrackerManager
from player_clustering import ClusteringManager
from tactical_analysis.homography import HomographyTransformer
from pipelines.tactical_pipeline import TacticalPipeline
from pipelines.keypoint_pipeline import KeypointPipeline
from player_annotations import AnnotatorManager
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration

# CSV header (v3-compatible)
CSV_HEADER = [
    "帧号", "球员ID",
    "画面X", "画面Y",
    "雷达X", "雷达Y",
    "真实X(米)", "真实Y(米)",
    "队伍",
]

# Team colors (BGR)
TEAM_COLORS = {0: (255, 0, 128), 1: (255, 100, 0)}   # purple, orange
BALL_COLOR = (0, 255, 255)   # yellow
REF_COLOR = (128, 128, 128)  # gray
KP_COLOR = (0, 255, 0)       # green for keypoints
KP_LINE_COLOR = (0, 200, 0)  # dark green for connections


def _build_tactical_frame(team1_pts, team2_pts, ball_pts, ref_pts, pitch_config, size=(350, 230)):
    """Build a tactical radar mini-map showing player positions on the pitch."""
    pitch = draw_pitch(pitch_config)

    def _to_px(pt, w, h):
        """Convert pitch coords (cm) → radar pixel coords."""
        x = int(pt[0] / 12000 * w)
        y = int(pt[1] / 7000 * h)
        return (x, y)

    h, w = pitch.shape[:2]
    for pt in team1_pts:
        if not np.isnan(pt).any():
            cv2.circle(pitch, _to_px(pt, w, h), 5, TEAM_COLORS[0], -1)
    for pt in team2_pts:
        if not np.isnan(pt).any():
            cv2.circle(pitch, _to_px(pt, w, h), 5, TEAM_COLORS[1], -1)
    for pt in ball_pts:
        if not np.isnan(pt).any():
            cv2.circle(pitch, _to_px(pt, w, h), 3, BALL_COLOR, -1)
    for pt in ref_pts:
        if not np.isnan(pt).any():
            cv2.circle(pitch, _to_px(pt, w, h), 3, REF_COLOR, -1)

    return cv2.resize(pitch, size)


def _draw_keypoints(frame, keypoints, connections=KEYPOINT_CONNECTIONS):
    """Draw 29 pitch keypoints and connections on the frame."""
    if keypoints is None or len(keypoints) == 0:
        return frame
    vis = frame.copy()
    kp = keypoints[0]  # (29, 3) → x, y, conf

    # Draw connections
    for i, j in connections:
        if kp[i, 2] > 0.5 and kp[j, 2] > 0.5:
            pt1 = (int(kp[i, 0]), int(kp[i, 1]))
            pt2 = (int(kp[j, 0]), int(kp[j, 1]))
            cv2.line(vis, pt1, pt2, KP_LINE_COLOR, 1)

    # Draw keypoints
    for i in range(len(kp)):
        if kp[i, 2] > 0.5:
            pt = (int(kp[i, 0]), int(kp[i, 1]))
            cv2.circle(vis, pt, 3, KP_COLOR, -1)

    return vis


def main():
    parser = argparse.ArgumentParser(
        description="BlueDream v3plus — Export pitch coordinates from soccer video"
    )
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output CSV path (auto if omitted)")
    parser.add_argument("--device", default="cpu", help="Device: cuda / cpu")
    parser.add_argument("--viz", default="none",
                        choices=["none", "bbox", "radar", "keypoints", "all"],
                        help="Video visualization mode (default: none)")
    parser.add_argument("--video", default=None,
                        help="Output video path (required if --viz != none)")
    parser.add_argument("--frame-count", type=int, default=-1, help="Max frames (-1 = all)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (1 = every frame)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"[ERROR] Video not found: {source}")
        sys.exit(1)

    do_viz = args.viz != "none"
    if do_viz and not args.video:
        print("[ERROR] --video <path> is required when --viz != none")
        sys.exit(1)

    # Output paths
    csv_path = Path(args.output) if args.output else source.parent / f"coords_{source.stem}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("BlueDream v3plus — Pitch Coordinate Exporter")
    print(f"  Source:  {source}")
    print(f"  CSV:     {csv_path}")
    print(f"  Device:  {args.device}")
    print(f"  Viz:     {args.viz}")
    if do_viz:
        print(f"  Video:   {args.video}")
    print()

    # ═══════════════════════════════════════════════════════
    # Phase 1: Load Models
    # ═══════════════════════════════════════════════════════
    print("[1/5] Loading models...")
    t0 = time.time()

    detection_model = load_detection_model(str(_default_model_path))
    keypoint_model = load_keypoint_model(str(_default_keypoint_path))

    tracker = TrackerManager()
    clustering = ClusteringManager()
    homography = HomographyTransformer()
    pitch_config = SoccerPitchConfiguration()
    annotator = AnnotatorManager() if do_viz else None

    print(f"  Models loaded in {time.time() - t0:.1f}s")

    # ═══════════════════════════════════════════════════════
    # Phase 2: Train Team Assignment
    # ═══════════════════════════════════════════════════════
    print("[2/5] Training team assignment models...")
    t0 = time.time()

    import supervision as sv

    cap = cv2.VideoCapture(str(source))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    train_end = min(total_frames, 120 * 24)
    frame_generator = sv.get_video_frames_generator(str(source), stride=12, end=train_end)
    crops = []
    for frame in tqdm(frame_generator, desc="  Collecting crops"):
        player_det, _, _ = get_detections(detection_model, frame)
        cropped = clustering.embedding_extractor.get_player_crops(frame, player_det)
        crops.extend(cropped)

    if len(crops) >= 2:
        clustering.train_clustering_models(crops)
        print(f"  Trained on {len(crops)} player crops in {time.time() - t0:.1f}s")
    else:
        print(f"  [WARN] Only {len(crops)} crops — team classification disabled")

    # ═══════════════════════════════════════════════════════
    # Phase 3: Frame-by-Frame Processing (streaming — no preload)
    # ═══════════════════════════════════════════════════════
    print("[3/5] Processing frames (detect + track + teams + homography)...")

    cap = cv2.VideoCapture(str(source))
    total_frames2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Apply stride and frame-count limit
    stride = args.stride
    frame_limit = total_frames2 if args.frame_count <= 0 else min(args.frame_count, total_frames2)
    n_process = (frame_limit + stride - 1) // stride
    out_fps = fps / stride if stride > 1 else fps
    print(f"  {n_process} frames to process @ {out_fps:.1f} fps (stride={stride})")

    # Video writer (init on first frame)
    vid_writer = None
    first_frame = True

    # CSV output
    csv_file = open(str(csv_path), "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)

    use_viz_bbox = args.viz in ("bbox", "all")
    use_viz_radar = args.viz in ("radar", "all")
    use_viz_kp = args.viz in ("keypoints", "all")

    # Homography temporal smoothing state (EMA to prevent jitter/drift)
    prev_H = None        # smoothed homography matrix
    prev_valid_H = None  # fallback when current frame fails

    raw_idx = 0
    proc_idx = 0
    pbar = tqdm(total=n_process, desc="  Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if raw_idx >= frame_limit:
            break
        raw_idx += 1

        # Apply stride
        if (raw_idx - 1) % stride != 0:
            continue

        proc_idx += 1
        frame_num = raw_idx
        pbar.update(1)

        # Init video writer on first frame
        if first_frame and do_viz:
            vid_writer = cv2.VideoWriter(
                args.video, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h)
            )
            first_frame = False

        # ── 3a. Detection ──
        player_det, ball_det, ref_det = get_detections(detection_model, frame)

        # ── 3b. Tracking ──
        player_det = tracker.process_tracking_for_frame(player_det)

        # ── 3c. Team assignment ──
        if len(player_det.xyxy) > 0 and len(crops) >= 2:
            labels = clustering.get_cluster_labels(frame, player_det)
            player_det.class_id = labels

        # ── 3d. Keypoint detection + Homography (with temporal smoothing) ──
        _, keypoints = get_keypoint_detections(keypoint_model, frame)
        view_transformer = homography.transform_to_pitch_keypoints(keypoints)

        # Apply EMA smoothing to homography matrix to eliminate jitter
        if view_transformer is not None:
            H_new = view_transformer.m.copy()
            if prev_H is not None:
                delta = np.linalg.norm(H_new - prev_H)
                if delta > 0.5:
                    # Large jump: heavy smoothing (90% old, 10% new)
                    alpha = 0.9
                else:
                    # Normal frame: light smoothing (70% old, 30% new)
                    alpha = 0.7
                H_smooth = alpha * prev_H + (1 - alpha) * H_new
                view_transformer.m = H_smooth
                prev_H = H_smooth
            else:
                prev_H = H_new
            prev_valid_H = prev_H
        elif prev_valid_H is not None:
            # Current frame failed — reuse last valid homography
            # Recreate a ViewTransformer with the cached matrix
            from sports.common.view import ViewTransformer
            view_transformer = ViewTransformer.__new__(ViewTransformer)
            view_transformer.m = prev_valid_H.copy()
            prev_H = prev_valid_H  # keep cached value

        # ── 3e. Transform to pitch coords ──
        all_players = []
        team1_pts, team2_pts, ball_pts_pitch, ref_pts_pitch = [], [], [], []

        if view_transformer is not None:
            # Player coords
            if len(player_det.xyxy) > 0:
                bboxes = player_det.xyxy
                center_points = np.array([
                    [(b[0] + b[2]) / 2, b[3]] for b in bboxes
                ], dtype=np.float32)
                try:
                    pitch_points = view_transformer.transform_points(points=center_points)
                    for i, (tid, bbox, (px, py), team_id) in enumerate(zip(
                        player_det.tracker_id, player_det.xyxy,
                        pitch_points, player_det.class_id,
                    )):
                        all_players.append({
                            "tracker_id": int(tid) if tid is not None else i,
                            "screen_x": (bbox[0] + bbox[2]) / 2,
                            "screen_y": bbox[3],
                            "pitch_x": float(px), "pitch_y": float(py),
                            "team": int(team_id),
                        })
                        if team_id == 0:
                            team1_pts.append((px, py))
                        else:
                            team2_pts.append((px, py))
                except Exception:
                    pass

            # Ball/referee coords
            if len(ball_det.xyxy) > 0:
                bc = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in ball_det.xyxy], dtype=np.float32)
                try:
                    bp = view_transformer.transform_points(points=bc)
                    ball_pts_pitch = [(float(b[0]), float(b[1])) for b in bp]
                except Exception:
                    pass
            if len(ref_det.xyxy) > 0:
                rc = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in ref_det.xyxy], dtype=np.float32)
                try:
                    rp = view_transformer.transform_points(points=rc)
                    ref_pts_pitch = [(float(r[0]), float(r[1])) for r in rp]
                except Exception:
                    pass

        # ── 3f. Write CSV ──
        # Homography maps to SoccerPitchConfiguration space (10500×6800 cm).
        # Divide by 100 to get meters, then apply boundary filtering:
        #   Soft clamp:  [-2, 107] × [-2,  70] m  (2m tolerance for sideline)
        #   Discard:    [-5, 110] × [-5,  73] m  (clear outlier → skip)
        PITCH_M = FIFA_FIELD_LENGTH   # 105.0
        PITCH_W = FIFA_FIELD_WIDTH    #  68.0
        CLAMP_MIN_X, CLAMP_MAX_X = -2.0, PITCH_M + 2.0
        CLAMP_MIN_Y, CLAMP_MAX_Y = -2.0, PITCH_W + 2.0
        DISCARD_MIN_X, DISCARD_MAX_X = -5.0, PITCH_M + 5.0
        DISCARD_MIN_Y, DISCARD_MAX_Y = -5.0, PITCH_W + 5.0

        for p in all_players:
            rx = p["pitch_x"] / 100.0  # cm → m
            ry = p["pitch_y"] / 100.0

            # Discard if clearly out of bounds (bad keypoint / matrix)
            if not (DISCARD_MIN_X <= rx <= DISCARD_MAX_X and
                    DISCARD_MIN_Y <= ry <= DISCARD_MAX_Y):
                continue  # skip this player — likely a detection error

            # Clamp to valid pitch area with 2m tolerance
            rx = max(CLAMP_MIN_X, min(CLAMP_MAX_X, rx))
            ry = max(CLAMP_MIN_Y, min(CLAMP_MAX_Y, ry))

            csv_writer.writerow([
                frame_num, p["tracker_id"],
                round(p["screen_x"], 1), round(p["screen_y"], 1),
                round(p["pitch_x"], 1), round(p["pitch_y"], 1),
                round(rx, 2), round(ry, 2), p["team"],
            ])

        # ── 3g. Video visualization ──
        if vid_writer is not None:
            output = frame.copy()

            # Bounding boxes
            if use_viz_bbox:
                output = annotator.annotate_all(output, player_det, ball_det, ref_det)

            # Keypoints
            if use_viz_kp and keypoints is not None:
                output = _draw_keypoints(output, keypoints)

            # Radar overlay
            if use_viz_radar and view_transformer is not None:
                radar = _build_tactical_frame(
                    team1_pts, team2_pts, ball_pts_pitch, ref_pts_pitch,
                    pitch_config, size=(350, 230)
                )
                rh, rw = radar.shape[:2]
                # Bottom-right overlay
                x, y = output.shape[1] - rw - 10, output.shape[0] - rh - 10
                output[y:y+rh, x:x+rw] = radar
                cv2.rectangle(output, (x-2, y-2), (x+rw+2, y+rh+2), (255, 255, 255), 2)
                cv2.putText(output, "Tactical", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            vid_writer.write(output)

    pbar.close()
    cap.release()
    csv_file.close()
    if vid_writer:
        vid_writer.release()

    print(f"\n[4/5] CSV saved: {csv_path}")
    print(f"  {proc_idx} frames processed")

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("[5/5] Done!")
    print(f"  Pitch: 105.0m × 68.0m (fixed standard)")
    if do_viz:
        print(f"  Viz video: {args.video}")


if __name__ == "__main__":
    main()
