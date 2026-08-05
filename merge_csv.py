#!/usr/bin/env python3
"""
BlueDream v3plus — CSV Merger

Merges per-segment detection CSVs (from batch_detect.py) into a single file
with continuous frame numbering.

Usage:
    # Merge all CSVs in a directory
    python merge_csv.py --dir C:\path\to\csvs --range all

    # Merge segments 0 through 10 (inclusive)
    python merge_csv.py --dir C:\path\to\csvs --range 0-10

    # Merge segments 5 through 15, custom output name
    python merge_csv.py --dir C:\path\to\csvs --range 5-15 --output merged.csv

Output:
    merged_coords.csv (or --output path)
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple


def extract_number(filename: str) -> int:
    """Extract the first integer from a filename for sorting.
    e.g. 'coords_segment_0001.csv' → 1, 'coords_segment_0123.csv' → 123
    """
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else 0


def find_csvs(directory: Path, pattern: str = "coords_*.csv") -> List[Path]:
    """Find CSV files, sorted by embedded number."""
    csvs = list(directory.glob(pattern))
    if not csvs:
        # Also try broader patterns
        csvs = list(directory.glob("*.csv"))
        # Exclude already-merged files
        csvs = [c for c in csvs if "merged" not in c.name.lower()]
    csvs.sort(key=lambda p: extract_number(p.name))
    return csvs


def filter_by_range(csvs: List[Path], range_spec: str) -> List[Path]:
    """Filter CSV list by range specification.

    Args:
        csvs: Sorted list of CSV paths
        range_spec: 'all' or 'N-M' (e.g. '0-10')

    Returns:
        Filtered CSV list
    """
    if range_spec.lower() == "all":
        return csvs

    try:
        parts = range_spec.split("-")
        if len(parts) != 2:
            raise ValueError
        start = int(parts[0].strip())
        end = int(parts[1].strip())
    except ValueError:
        print(f"[ERROR] Invalid range: '{range_spec}'. Use 'all' or 'N-M' (e.g. '0-10')")
        sys.exit(1)

    filtered = [c for c in csvs if start <= extract_number(c.name) <= end]
    if not filtered:
        print(f"[WARN] No CSV files found in range {start}-{end}")
    return filtered


def merge_csvs(csv_paths: List[Path], output_path: Path):
    """Merge CSVs with sequential frame numbering.

    Each segment's frames start from 0 or 1 internally. We offset each
    subsequent segment by the cumulative frame count so the merged file
    has continuous numbering.
    """
    if not csv_paths:
        print("[ERROR] No CSV files to merge.")
        sys.exit(1)

    print(f"Merging {len(csv_paths)} CSV files:")
    for i, cp in enumerate(csv_paths):
        print(f"  [{i}] {cp.name}")

    all_rows: List[dict] = []
    fieldnames = None
    global_offset = 0
    segment_stats: List[Tuple[str, int, int]] = []  # (name, frames, players)

    for i, cp in enumerate(csv_paths):
        rows = []
        with open(cp, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)

        if not rows:
            print(f"  [WARN] {cp.name} is empty — skipping")
            segment_stats.append((cp.name, 0, 0))
            continue

        # Find max frame in this segment (for offset calculation)
        seg_max = max(int(r["帧号"]) for r in rows)
        seg_min = min(int(r["帧号"]) for r in rows)
        n_players = len(set(r["球员ID"] for r in rows))

        # Apply offset and collect
        for row in rows:
            row["帧号"] = str(int(row["帧号"]) + global_offset)
            all_rows.append(row)

        segment_stats.append((cp.name, len(rows), n_players))
        global_offset += (seg_max - seg_min + 1)  # next segment starts after this one

    # Write merged CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    print(f"\nMerged: {output_path}")
    print(f"  Total rows:  {len(all_rows)}")
    print(f"  Total frames: {global_offset}")
    print(f"\nPer-segment breakdown:")
    for name, rows, players in segment_stats:
        print(f"  {name}: {rows} rows, {players} unique players")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="BlueDream v3plus — Merge batch_detect.py CSV outputs"
    )
    parser.add_argument(
        "--dir", required=True,
        help="Directory containing per-segment CSV files"
    )
    parser.add_argument(
        "--range", default="all",
        help="Segment range: 'all' (default) or 'N-M' (e.g. '0-10')"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: <dir>/merged_coords.csv)"
    )
    parser.add_argument(
        "--pattern", default="coords_*.csv",
        help="CSV filename glob (default: coords_*.csv)"
    )
    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.exists():
        print(f"[ERROR] Directory not found: {directory}")
        sys.exit(1)

    # Find and filter CSVs
    csvs = find_csvs(directory, args.pattern)
    if not csvs:
        print(f"[ERROR] No CSV files found in {directory} (pattern: {args.pattern})")
        sys.exit(1)

    print(f"Found {len(csvs)} CSV files in {directory}")
    csvs = filter_by_range(csvs, args.range)

    # Determine output path
    output = Path(args.output) if args.output else directory / "merged_coords.csv"

    merge_csvs(csvs, output)


if __name__ == "__main__":
    main()
