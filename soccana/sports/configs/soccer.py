"""Compatibility shim: sports.configs.soccer → detector.config"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # BlueDream_v3plus/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from detector.config import SoccerPitchConfiguration  # noqa: E402
