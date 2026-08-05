"""Compatibility shim: sports.annotators.soccer → detector.annotators"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # BlueDream_v3plus/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from detector.annotators import draw_pitch, draw_points_on_pitch  # noqa: E402
