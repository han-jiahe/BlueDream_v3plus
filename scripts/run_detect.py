#!/usr/bin/env python3
"""
BlueDream v3plus — Single Video Detection (soccana pipeline)

Thin wrapper around export_coords.py for backward-compatible CLI.
Equivalent to: python scripts/export_coords.py --source <video> ...

Usage:
    python scripts/run_detect.py --source video.mp4 --device cuda
    python scripts/run_detect.py --source video.mp4 --device cuda --video out.mp4
    python scripts/run_detect.py --source video.mp4 --output result.csv --viz all
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Delegate to export_coords
from export_coords import main

if __name__ == "__main__":
    main()
