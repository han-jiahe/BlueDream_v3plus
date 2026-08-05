"""
BlueDream v3plus — One-Click Launch
Usage: python main.py

Starts the FastAPI web server on http://127.0.0.1:8000
"""

import sys
import webbrowser
from pathlib import Path

# Ensure v3plus modules are importable
# IMPORTANT: web_app/ must come FIRST so `from config import`
# resolves to web_app/config.py, not detector/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_APP_DIR = Path(__file__).resolve().parent

# Insert module dirs first (order: append to front)
sys.path.insert(0, str(PROJECT_ROOT / "data_tools"))
sys.path.insert(0, str(PROJECT_ROOT / "vi_analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "detector"))
sys.path.insert(0, str(PROJECT_ROOT))
# web_app/ goes last → stays at index 0 (highest priority)
sys.path.insert(0, str(WEB_APP_DIR))

import uvicorn

if __name__ == "__main__":
    print(f"""
    ╔══════════════════════════════════════════╗
    ║       BlueDream v3plus                   ║
    ║  足球视频分析 Web 原型系统               ║
    ║  YOLOv11n soccana + Voronoi VI           ║
    ║  http://127.0.0.1:8000                   ║
    ╚══════════════════════════════════════════╝
    """)
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
