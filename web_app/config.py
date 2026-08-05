"""
BlueDream v3plus — Web App Configuration
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # BlueDream_v3plus/

# Module paths (v3plus architecture)
DETECTOR_DIR = PROJECT_ROOT / "detector"
VI_ANALYSIS_DIR = PROJECT_ROOT / "vi_analysis"
DATA_TOOLS_DIR = PROJECT_ROOT / "data_tools"

# Storage
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Analysis parameters
FRAMES_PER_SEGMENT = 120
PITCH_LENGTH = 105.0   # meters (FIFA standard)
PITCH_WIDTH = 68.0     # meters (FIFA standard)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ts", ".csv"}
