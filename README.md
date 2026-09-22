<div align="center">

  <h1>⚽ Innovative Entropy Output of Football Matches based on General Visual Recognition </h1>

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.2-9400D3)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)](https://fastapi.tiangolo.com/)

</div>

> **Version**: 3.0++  
> **Goal**: Build an open-source football match $\dot{VI}$ (**Variation of Information Rate**) analysis system based on the complete [Adit-jain/Soccer_Analysis](https://github.com/Adit-jain/Soccer_Analysis) source architecture, using YOLOv11n soccana + SigLIP team classification + 29-keypoint homography transformation.

---

## 📖 Introduction

BlueDream v3plus is an open-source football video analysis system. Given a match video, it automatically:

1. **Object Detection** — YOLOv11n soccana detects players, ball, and referees
2. **Multi-Object Tracking** — ByteTrack assigns a unique ID to each player across frames
3. **Team Classification** — SigLIP visual embeddings + UMAP dimensionality reduction + KMeans clustering to distinguish the two teams
4. **Pitch Registration** — 29 keypoint detection → homography → screen coordinates mapped to 105×68m real-world coordinates
5. **$\dot{VI}$ Analysis** — Voronoi tessellation and information-theoretic influence computation, quantifying each player's spatial control
6. **Visualization** — Heatmaps, trajectory plots, Voronoi diagrams, tactical radar overlays

Output is a standard CSV format, with an interactive FastAPI web interface for analysis.

---

## 🏗️ Project Architecture

```
BlueDream_v3plus/
├── soccana/                    ← Full Soccer_Analysis source (core pipeline)
│   ├── player_detection/       ← YOLOv11n soccana → Player / Ball / Referee
│   ├── player_tracking/        ← ByteTrack multi-object tracking
│   ├── player_clustering/      ← SigLIP + UMAP + KMeans team assignment
│   ├── keypoint_detection/     ← YOLOv11n-pose → 29 pitch keypoints
│   ├── tactical_analysis/      ← Homography → real-world coordinate mapping
│   ├── pipelines/              ← Pipeline coordination layer
│   └── sports/                 ← Compatibility shim (redirects to detector/, no roboflow dep)
│
├── detector/                   ← Self-implemented base components (replaces roboflow/sports)
│   ├── config.py               ← SoccerPitchConfiguration (105×68m)
│   ├── view_transformer.py     ← Perspective transform
│   └── annotators.py           ← Pitch/radar drawing
│
├── scripts/
│   ├── export_coords.py        ← ★ Core: video → CSV real-world coordinates
│   ├── run_detect.py           ← CLI entry point
│   └── batch_detect.py         ← Multi-segment concurrency (ModelArts)
│
├── merge_csv.py                ← Merge multi-segment CSVs
├── vi_analysis/                ← Voronoi VI + Clustering VI
├── data_tools/                 ← CSV → JSON conversion
├── web_app/                    ← FastAPI Web
├── models/                     ← soccana.pt + soccana_keypoint.pt
└── docs/                       ← Deployment documentation
```


---

## 🚀 Quick Start

### Prerequisites
- #### Environment : contain Python (PyTorch) and Git. 
<details>
<summary>Specific environment requirements are detailed below.</summary>

| Requirement | Recommended | Notes |
|------------|---------|-------|
| **Python** | 3.10+ | Conda or venv recommended for isolation |
| **Git** | 2.0+ | Required for cloning the repository |
| **NVIDIA GPU** | 6+ GB VRAM | YOLOv11n uses ~1.5 GB; CPU-only mode also supported (`--device cpu`) |
| **CUDA** | 11.8+ | Must match your PyTorch version |
| **Disk Space** | 15+ GB | Models (~6 MB) + dependencies + video storage |
| **RAM** | 9+ GB | For loading full-match video frames |
| **OS** | Windows 10+ / Linux | macOS supported for Web/VI analysis only (no CUDA) |

> **CPU-only usage**: If no GPU is available, use `--device cpu` for detection. Expect ~5–10× slower inference but identical output quality. VI analysis and Web service do not require GPU at all.

</details>

- #### Models : obtain `soccana.pt` (object detection) and `soccana_keypoint.pt` (soccer field keypoint detection)
> I. You can run the command **`python scripts/download_models.py`** to download the pre-trained models to the `models/` from HuggingFace automatically.  
> II. If you want to train models in person, please refer to [🔗 Models and Datasets](https://github.com/han-jiahe/BlueDream_v3plus/tree/main#-models--datasets-), remember to rename the trained weight files and place them in the `models/`.


### 1. Local Web (Chinese version)

```bash
cd C:\Users\<your-name>
git clone https://github.com/han-jiahe/BlueDream_v3plus.git
cd C:\Users\<your-name>\BlueDream_v3plus\web_app
pip install -r requirements.txt
python main.py
# Opens http://127.0.0.1:8000
# Upload the videos, after which they will be placed in C:\Users\<your-name>\BlueDream_v3plus\web_app\uploads
```

### 2.1. Single Video Detection
Recommended: run locally (Windows, GPU)

> To avoid version conflicts, You'd better build a virtual environment locally by yourself such as Python venv (lightweight) or others.
```powershell
# After activating your virtual environment
cd C:\Users\<your-name>\BlueDream_v3plus
pip install -r requirements.txt

# Detect + output CSV + radar video
python scripts/run_detect.py --source web_app\uploads\<video> --device cuda --viz radar --video web_app\uploads\out.mp4
```

| Scenario | Location |
|----------|----------|
| **Single video** | Local GPU (YOLOv11n lightweight, ~1.5GB VRAM) |
| **Multi-segment batch (10+)** | ModelArts Tesla T4 (¥14/hr), `batch_detect.py --workers 4` |

### 2.2. Batch Detection of Video Segments
ModelArts Notebook (rent GPU for concurrent detection)
> Ensure you have completed project cloning, conda environment setup, and model download per [docs/DEPLOYMENT_MODELARTS.md](docs/DEPLOYMENT_MODELARTS.md). The following are daily usage commands. 
 ( **If you have no less than 12 GB of video memory locally, you can also perform batch detection locally.** Since the Huawei Cloud platform is set to Chinese by default, <font color="red">the manual [docs/DEPLOYMENT_MODELARTS.md](docs/DEPLOYMENT_MODELARTS.md) is also in Chinese</font>.)


After a Notebook stop, the environment is lost. Use `requirements-modelarts.txt` to quickly rebuild:

```bash
conda create -n soccer_v3plus python=3.10 -y && conda activate soccer_v3plus
pip install -r requirements-modelarts.txt
export HF_ENDPOINT=https://hf-mirror.com

# Upload multiple segments to /path/to/videos, run batch detection
python scripts/batch_detect.py --segments_dir /path/to/videos --workers 4 --device cuda

# If the segments come from a single video, merge them
python merge_csv.py --dir /path/to/csvs --range all
```

**Merge script parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--dir` | Yes | — | Directory containing CSV files |
| `--range` | No | `all` | Merge range: `all` or `N-M` (e.g. `0-10`) |
| `--output` | No | `merged_coords.csv` | Merged output CSV path |
| `--pattern` | No | `coords_*.csv` | CSV filename glob pattern |

### 3. Local Web Data Analysis
Return to http://127.0.0.1:8000, upload CSV files and run analysis. If batch detection was used, some features are limited:
- Merged CSV: a single player may not map to a consistent ID across segments; only team-level and match-level statistics are reliable. Player-level analysis is not available.
- Individual segment CSV: all data is valid, but due to short duration, match-level statistics may be too random to be meaningful.

**<font color="red">Note:</font>** For detailed workflows and FAQs on match video detection, see [docs/DEPLOYMENT_MODELARTS.md](docs/DEPLOYMENT_MODELARTS.md).

---

## 📊 Analysis Modes

The system supports **modular analysis** — you don't need to run everything at once:

| Analysis Type | Description | Extra Input |
|--------------|-------------|-------------|
| **Full Analysis** | $\dot{VI}$ distribution timeline + density curve + player $\dot{VI}$ ranking + Top 3 player heatmaps | — |
| **Single Player** | $\dot{VI}$ timeseries + team average comparison + personal heatmap | Player ID |
| **Trajectory** | Multi-player trajectory on 105×68m standard pitch **("+ ball" function is currently not supported, stay tuned)** | Player IDs (comma-separated) |
| **Voronoi Viewer** | Voronoi tessellation at a given match percentage (e.g. 50%) | Percentage (0-100) |
| **$$\dot{VI}$$ Distribution** | $\dot{VI}$ timeline + density curve + ranking (no heatmaps) | — |
| **Heatmap Only** | Player activity heatmap on the pitch | Player ID |

Each analysis mode runs independently, generating corresponding PNG charts for download.

---

## 🛠️ Detection CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source` | (required) | Input video path |
| `--device` | `cpu` | `cuda` / `cpu` |
| `--output` | Auto | CSV output path |
| `--viz` | `none` | Video visualization: `none` / `bbox` / `radar` / `keypoints` / `all` |
| `--video` | — | Visualization video output (required if `--viz ≠ none`) |
| `--frame-count` | `-1` | Max frames to process (`-1` = all) |
| `--stride` | `1` | Frame interval |

- **`--viz` modes**:

| Value | Output |
|-------|--------|
| `none` | CSV only (fastest) |
| `bbox` | Player boxes + IDs + team colors |
| `radar` | Tactical radar overlay (bottom-right) |
| `keypoints` | 29 keypoints + connection lines |
| `all` | All three combined |

---

## 📋 Dependency Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Full local dependencies (detection + Web + $\dot{VI}$ analysis) |
| `requirements-modelarts.txt` | Minimal Notebook dependencies (detection only, no Web/VI) — for quick rebuild after stop |
| `web_app/requirements.txt` | Web-specific dependencies (FastAPI + visualization) |

---

## 🔗 Models & Datasets <a id="Models"></a>

### Pre-trained Models

| Model Type | HuggingFace Repository | Description |
|------------|----------------------|-------------|
| **Object Detection** | [Adit-jain/soccana](https://huggingface.co/Adit-jain/soccana) | YOLOv11n for player, ball, and referee detection |
| **Keypoint Detection** | [Adit-jain/Soccana_Keypoint](https://huggingface.co/Adit-jain/Soccana_Keypoint) | YOLOv11n-pose for 29-point pitch keypoint detection |

### Training Datasets

| Dataset Type | HuggingFace Repository | Description |
|--------------|----------------------|-------------|
| **Keypoint Detection** | [Adit-jain/Soccana_Keypoint_detection_v1](https://huggingface.co/datasets/Adit-jain/Soccana_Keypoint_detection_v1) | Annotated dataset with 29 field reference points |
| **Object Detection** | [Adit-jain/Soccana_player_ball_detection_v1](https://huggingface.co/datasets/Adit-jain/Soccana_player_ball_detection_v1) | Player, ball, and referee detection dataset |

---

## 🙏 Acknowledgments

This project is built upon the following open-source projects and research:

- **Core Detection Architecture**: [Adit-jain/Soccer_Analysis](https://github.com/Adit-jain/Soccer_Analysis) — complete football video analysis pipeline providing YOLOv11n detection, ByteTrack tracking, SigLIP team classification, and 29-keypoint pitch registration
- **Pre-trained Models**: [Adit-jain/soccana](https://huggingface.co/Adit-jain/soccana) and [Adit-jain/Soccana_Keypoint](https://huggingface.co/Adit-jain/Soccana_Keypoint) — trained and open-sourced by Adit Jain
- **$\dot{VI}$ Analysis Methodology**: Information-theoretic football analysis research from https://doi.org/10.48550/arXiv.2102.11182 by L.R. Pereira, R.J. Lopes, J. Luka, D. Araujo, J. Ramos
- **Foundation Libraries**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [Supervision](https://github.com/roboflow/supervision), [FastAPI](https://fastapi.tiangolo.com/)
