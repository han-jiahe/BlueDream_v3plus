"""
BlueDream v3plus — Annotators
Drawing utilities for pitch, radar view, and player overlays.
Self-contained — no roboflow/sports dependency.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from .config import SoccerPitchConfiguration


# ── Default Colors ──────────────────────────────────────

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']  # team0, team1, ref, ball
PLAYER_COLORS = ['#1E90FF', '#FF6347']                   # home, away
BALL_COLOR = '#FF0000'
REF_COLOR = '#FFD700'

# Supervision annotators for video overlay
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2,
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2,
)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)


# ── Pitch Drawing ───────────────────────────────────────

def _cm_to_meters(vertices_cm: List[Tuple[int, int]]) -> np.ndarray:
    """Convert vertex coordinates from cm to meters."""
    return np.array(vertices_cm, dtype=np.float32) / 100.0


def draw_pitch(
    config: SoccerPitchConfiguration = None,
    background_color: sv.Color = None,
    line_color: sv.Color = None,
    padding: int = 50,
    line_thickness: int = 2,
    point_radius: int = 4,
    scale: float = 0.1,
) -> np.ndarray:
    """
    Draw a top-down soccer pitch.

    Args:
        config: Pitch configuration
        background_color: Background color (default: green)
        line_color: Line color (default: white)
        padding: Border padding in pixels
        line_thickness: Line thickness in pixels
        point_radius: Keypoint dot radius
        scale: Pixels per cm (0.1 = 1m → 100px)

    Returns:
        BGR pitch image as numpy array
    """
    config = config or SoccerPitchConfiguration()
    bg = background_color or sv.Color.from_hex('#2E7D32')  # grass green
    fg = line_color or sv.Color.from_hex('#FFFFFF')

    # Image dimensions
    w = int(config.length * scale) + 2 * padding
    h = int(config.width * scale) + 2 * padding
    img = np.full((h, w, 3), bg.as_bgr(), dtype=np.uint8)

    # Draw edges (pitch marking lines)
    vertices_m = _cm_to_meters(config.vertices)
    for i, j in config.edges:
        pt1 = _to_pixel(vertices_m[i - 1], scale, padding)
        pt2 = _to_pixel(vertices_m[j - 1], scale, padding)
        cv2.line(img, pt1, pt2, fg.as_bgr(), line_thickness)

    # Centre circle
    center_x = config.length / 2 * scale + padding
    center_y = config.width / 2 * scale + padding
    radius = int(config.centre_circle_radius * scale)
    cv2.circle(img, (int(center_x), int(center_y)), radius, fg.as_bgr(), line_thickness)

    # Penalty spots
    spot_left = (int(config.penalty_spot_distance * scale + padding),
                 int(config.width / 2 * scale + padding))
    spot_right = (int((config.length - config.penalty_spot_distance) * scale + padding),
                  int(config.width / 2 * scale + padding))
    cv2.circle(img, spot_left, point_radius, fg.as_bgr(), -1)
    cv2.circle(img, spot_right, point_radius, fg.as_bgr(), -1)

    return img


def _to_pixel(pt_meters: np.ndarray, scale: float, padding: int) -> Tuple[int, int]:
    return int(pt_meters[0] * scale * 100 + padding), int(pt_meters[1] * scale * 100 + padding)


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color,
    edge_color: sv.Color = None,
    radius: int = 12,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: np.ndarray = None,
) -> np.ndarray:
    """
    Draw player positions as colored dots on a pitch image.

    Args:
        config: Pitch configuration
        xy: (N, 2) player coordinates in cm
        face_color: Fill color
        edge_color: Outline color (default: white)
        radius: Dot radius
        thickness: Edge thickness
        padding: Same as draw_pitch
        scale: Same as draw_pitch
        pitch: Existing pitch image (created if None)

    Returns:
        Annotated pitch image
    """
    if pitch is None:
        pitch = draw_pitch(config, padding=padding, scale=scale)
    if edge_color is None:
        edge_color = sv.Color.from_hex('#FFFFFF')
    if xy is None or len(xy) == 0:
        return pitch

    for pt in xy:
        px, py = _to_pixel(np.array(pt) / 100.0, scale, padding)
        cv2.circle(pitch, (px, py), radius, face_color.as_bgr(), -1)
        cv2.circle(pitch, (px, py), radius, edge_color.as_bgr(), thickness)

    return pitch


# ── Video Overlay Helpers ───────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    color_lookup: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    use_ellipse: bool = True,
) -> np.ndarray:
    """
    Annotate a video frame with player detections.

    Args:
        frame: BGR video frame
        detections: supervision Detections
        color_lookup: (N,) custom color indices per detection
        labels: (N,) label strings per detection
        use_ellipse: Use ellipse (True) or box (False) for player markers

    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    if color_lookup is not None:
        if use_ellipse:
            annotated = ELLIPSE_ANNOTATOR.annotate(
                annotated, detections, custom_color_lookup=color_lookup
            )
        else:
            annotated = BOX_ANNOTATOR.annotate(
                annotated, detections, custom_color_lookup=color_lookup
            )
        if labels:
            annotated = LABEL_ANNOTATOR.annotate(
                annotated, detections, labels, custom_color_lookup=color_lookup
            )
    else:
        if use_ellipse:
            annotated = ELLIPSE_ANNOTATOR.annotate(annotated, detections)
        else:
            annotated = BOX_ANNOTATOR.annotate(annotated, detections)
    return annotated


def overlay_radar(
    frame: np.ndarray,
    radar: np.ndarray,
    position: str = "bottom",
    opacity: float = 0.5,
) -> np.ndarray:
    """
    Overlay a radar mini-map onto the video frame.

    Args:
        frame: Video frame
        radar: Radar pitch image (smaller than frame)
        position: "bottom" or "bottom-right"
        opacity: Blend opacity (0=transparent, 1=opaque)

    Returns:
        Frame with radar overlay
    """
    h, w = frame.shape[:2]
    rh, rw = radar.shape[:2]

    if position == "bottom":
        x = w // 2 - rw // 2
        y = h - rh
    else:
        x = w - rw - 20
        y = h - rh - 20

    # Ensure radar fits
    x = max(0, min(x, w - rw))
    y = max(0, min(y, h - rh))

    roi = frame[y:y + rh, x:x + rw]
    blended = cv2.addWeighted(roi, 1 - opacity, radar, opacity, 0)
    frame[y:y + rh, x:x + rw] = blended
    return frame
