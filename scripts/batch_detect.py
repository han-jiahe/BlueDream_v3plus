#!/usr/bin/env python3
"""
BlueDream v3plus — GPU Batch Detection (Multi-Segment)

Processes multiple video segments using the full soccana pipeline.
Each segment gets its own worker subprocess.

Usage:
    python scripts/batch_detect.py --segments_dir /path/to/segments --workers 3 --device cuda

Output:
    One CSV per video segment: coords_<segment_name>.csv
    Format: 帧号,球员ID,画面X,画面Y,雷达X,雷达Y,真实X(米),真实Y(米),队伍
"""

import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_SCRIPT = Path(__file__).resolve().parent / "export_coords.py"


def process_segment(video_path: str, csv_path: str, device: str):
    """Run export_coords.py on a single video segment."""
    cmd = [
        sys.executable, str(EXPORT_SCRIPT),
        "--source", video_path,
        "--output", csv_path,
        "--device", device,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"export_coords failed:\n{result.stderr}")
    return result.stdout


class SegmentWorker(Process):
    """Independent process: run soccana pipeline on one video segment."""

    def __init__(self, task_queue, result_queue, device="cuda", gpu_id=0):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.device = device
        self.gpu_id = gpu_id

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        while True:
            task = self.task_queue.get()
            if task is None:
                break

            video_path, csv_output = task
            csv_path = Path(csv_output)
            if csv_path.exists() and csv_path.stat().st_size > 500:
                self.result_queue.put(("skip", str(video_path), str(csv_output)))
                continue

            try:
                process_segment(video_path, csv_output, self.device)
                self.result_queue.put(("ok", str(video_path), str(csv_output)))
            except Exception as e:
                self.result_queue.put(("error", str(video_path), str(e)))


def main():
    parser = argparse.ArgumentParser(description="BlueDream v3plus — GPU Batch Detection")
    parser.add_argument("--segments_dir", required=True, help="Video segments directory")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent workers (default: 2, 32GB VRAM: 3-4)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output_dir", default=None, help="CSV output directory")
    parser.add_argument("--pattern", default="segment_*.mp4",
                        help="Video filename glob (default: segment_*.mp4)")
    args = parser.parse_args()

    seg_dir = Path(args.segments_dir)
    if not seg_dir.exists():
        print(f"[ERROR] Directory not found: {seg_dir}")
        sys.exit(1)

    videos = sorted(seg_dir.glob(args.pattern))
    if not videos:
        print(f"[ERROR] No videos matching '{args.pattern}' in {seg_dir}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else seg_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare tasks
    tasks = []
    for v in videos:
        csv_path = out_dir / f"coords_{v.stem}.csv"
        if csv_path.exists() and csv_path.stat().st_size > 500:
            print(f"[SKIP] {v.name} -> {csv_path.name}")
            continue
        tasks.append((str(v), str(csv_path)))

    if not tasks:
        print("All segments already have results. Nothing to do.")
        return

    n_workers = min(args.workers, len(tasks))
    print(f"\nPending: {len(tasks)} segments | Workers: {n_workers} | GPU: {args.device}")
    print(f"Output dir: {out_dir}\n")

    task_queue = Queue()
    result_queue = Queue()

    # Start workers
    workers = []
    for i in range(n_workers):
        w = SegmentWorker(task_queue, result_queue, device=args.device, gpu_id=i)
        w.start()
        workers.append(w)

    # Submit tasks
    for t in tasks:
        task_queue.put(t)
    for _ in range(n_workers):
        task_queue.put(None)

    # Collect results
    t0 = time.time()
    completed = skipped = errors = 0
    for _ in range(len(tasks)):
        status, path, msg = result_queue.get()
        if status == "ok":
            completed += 1
            print(f"[OK] {Path(path).name} ({completed}/{len(tasks)})")
        elif status == "skip":
            skipped += 1
        elif status == "error":
            errors += 1
            print(f"[ERR] {Path(path).name}: {msg}")

    for w in workers:
        w.join()

    elapsed = time.time() - t0
    print(f"\nDone! {completed} OK, {skipped} skipped, {errors} errors | "
          f"Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
