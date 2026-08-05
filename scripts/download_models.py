#!/usr/bin/env python3
"""
BlueDream v3plus — Download Models

Downloads required model files:
  1. YOLOv11n soccana (from HuggingFace: Adit-jain/soccana)
  2. Football pitch keypoint detection model

Usage:
    python scripts/download_models.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def download_soccana():
    """
    Download YOLOv11n soccana model from HuggingFace.

    Model source: https://huggingface.co/Adit-jain/soccana
    File path:    Model/weights/best.pt
    hf-mirror:    https://hf-mirror.com/Adit-jain/soccana/resolve/main/Model/weights/best.pt
    """
    dest = MODELS_DIR / "soccana.pt"
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[SKIP] soccana.pt already exists ({size_mb:.1f} MB)")
        return

    print("Downloading YOLOv11n soccana (Player/Ball/Referee, 3 classes)...")

    url_mirror = "https://hf-mirror.com/Adit-jain/soccana/resolve/main/Model/weights/best.pt"
    url_direct = "https://huggingface.co/Adit-jain/soccana/resolve/main/Model/weights/best.pt"

    # Try hf-mirror first (China access)
    try:
        import urllib.request
        print(f"  Mirror: {url_mirror}")
        req = urllib.request.Request(url_mirror, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status == 200:
                with open(dest, "wb") as f:
                    f.write(resp.read())
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  [OK] soccana.pt downloaded ({size_mb:.1f} MB)")
                return
    except Exception as e:
        print(f"  hf-mirror failed: {e}")

    # Try direct HuggingFace
    try:
        import urllib.request
        print(f"  Direct: {url_direct}")
        req = urllib.request.Request(url_direct, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status == 200:
                with open(dest, "wb") as f:
                    f.write(resp.read())
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  [OK] soccana.pt downloaded ({size_mb:.1f} MB)")
                return
    except Exception as e:
        print(f"  direct failed: {e}")

    print(f"\n[INFO] Automatic download failed. Please manually download:")
    print(f"  1. Open: https://hf-mirror.com/Adit-jain/soccana/tree/main/Model/weights")
    print(f"  2. Download best.pt")
    print(f"  3. Save as: {dest}")
    print(f"\n  Or via VPN: https://huggingface.co/Adit-jain/soccana/tree/main/Model/weights")


def download_pitch_model():
    """
    Download pitch keypoint detection model (29 keypoints, YOLOv11n-pose).

    Tries hf-mirror first (China access), then direct HuggingFace.
    Falls back to legacy model if already present locally.
    """
    dest = MODELS_DIR / "soccana_keypoint.pt"
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[SKIP] soccana_keypoint.pt already exists ({size_mb:.1f} MB)")
        return

    print("Downloading Soccana_Keypoint (29 keypoints, YOLOv11n-pose)...")

    # Try hf-mirror (China access)
    url = "https://hf-mirror.com/Adit-jain/Soccana_Keypoint/resolve/main/Model/weights/best.pt"
    try:
        import urllib.request
        print(f"  Mirror: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status == 200:
                with open(dest, "wb") as f:
                    f.write(resp.read())
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  [OK] soccana_keypoint.pt downloaded ({size_mb:.1f} MB)")
                return
    except Exception as e:
        print(f"  hf-mirror failed: {e}")

    # Try direct HuggingFace
    url = "https://huggingface.co/Adit-jain/Soccana_Keypoint/resolve/main/Model/weights/best.pt"
    try:
        import urllib.request
        print(f"  Direct: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status == 200:
                with open(dest, "wb") as f:
                    f.write(resp.read())
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  [OK] soccana_keypoint.pt downloaded ({size_mb:.1f} MB)")
                return
    except Exception as e:
        print(f"  direct failed: {e}")

    # Fallback: check for legacy model
    legacy = MODELS_DIR / "football-pitch-detection.pt"
    if legacy.exists():
        print(f"  [INFO] Using legacy football-pitch-detection.pt instead")
        print(f"  [INFO] Rename to soccana_keypoint.pt or set up Soccana_Keypoint download")
        return

    print(f"\n[INFO] Automatic download failed. Please manually download:")
    print(f"  1. Open: https://hf-mirror.com/Adit-jain/Soccana_Keypoint/tree/main/Model/weights")
    print(f"  2. Download best.pt")
    print(f"  3. Save as: {dest}")
    print(f"\n  Or via VPN: https://huggingface.co/Adit-jain/Soccana_Keypoint/tree/main/Model/weights")


def main():
    print("BlueDream v3plus — Model Download")
    print(f"Models directory: {MODELS_DIR}\n")

    # 1. Soccana model
    download_soccana()

    # 2. Pitch model
    download_pitch_model()

    # Verify
    print("\nModel files:")
    for name in ["soccana.pt", "soccana_keypoint.pt"]:
        p = MODELS_DIR / name
        if p.exists():
            print(f"  [OK] {name} ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  [MISSING] {name}")

    print("\nDone. Models are ready for GPU detection.")


if __name__ == "__main__":
    main()
