<div align="center">

  <h1>⚽ 基于通用视觉识别的足球比赛创新性熵值产出</h1>

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.2-9400D3)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)](https://fastapi.tiangolo.com/)

</div>

> **版本**: v3.0++  
> **目标**: 基于 [Adit-jain/Soccer_Analysis](https://github.com/Adit-jain/Soccer_Analysis) 完整源码架构，融合 YOLOv11n soccana 目标检测、SigLIP 球队分类及 29 关键点单应性变换，构建开源足球比赛 $\dot{VI}$（信息变差率，Variation of Information Rate）分析系统。

---

## 📖 引言

BlueDream v3plus 是一个开源的足球比赛视频分析系统。给定一段比赛视频，系统自动完成以下流程：

1. **目标检测** — YOLOv11n soccana 模型检测球员、足球及裁判，返回三类目标的边界框与置信度
2. **多目标追踪** — ByteTrack 算法为每名球员分配跨帧一致的身份标识（tracker ID）
3. **球队分类** — SigLIP 视觉编码器提取球员图像嵌入 → UMAP 流形降维 → KMeans 聚类，自动区分两队队服
4. **边界对齐** — YOLOv11n-pose 检测 29 个球场关键点 → 单应性矩阵估计 → 将画面像素坐标映射至 105×68 m 的真实球场坐标系
5. **$\dot{VI}$ 分析** — 基于 Voronoi 空间剖分与信息论的信息变差率计算，量化每名球员的空间控制力及其动态演化
6. **可视化** — 输出热力图、运动轨迹、Voronoi 剖分多边形及战术雷达叠加视图

系统输出为标准 CSV 格式，并提供基于 FastAPI 的交互式 Web 分析界面。

---

## 🏗️ 项目架构

```
BlueDream_v3plus/
├── soccana/                    ← Soccer_Analysis 完整源码（核心管线）
│   ├── player_detection/       ← YOLOv11n soccana → 球员 / 足球 / 裁判
│   ├── player_tracking/        ← ByteTrack 多目标追踪
│   ├── player_clustering/      ← SigLIP + UMAP + KMeans 球队分类
│   ├── keypoint_detection/     ← YOLOv11n-pose → 29 球场关键点
│   ├── tactical_analysis/      ← 单应性变换 → 真实坐标映射
│   ├── pipelines/              ← 管线协调层
│   └── sports/                 ← 兼容层（重定向至 detector/，无 roboflow 依赖）
│
├── detector/                   ← 自实现基础组件（替代 roboflow/sports）
│   ├── config.py               ← SoccerPitchConfiguration（105×68 m）
│   ├── view_transformer.py     ← 透视变换
│   └── annotators.py           ← 球场/雷达绘制
│
├── scripts/
│   ├── export_coords.py        ← ★ 核心模块：视频 → CSV 真实坐标
│   ├── run_detect.py           ← CLI 入口
│   └── batch_detect.py         ← 多段并发检测（ModelArts）
│
├── merge_csv.py                ← 多段 CSV 拼接工具
├── vi_analysis/                ← Voronoi VI + 聚类 VI
├── data_tools/                 ← CSV → JSON 格式转换
├── web_app/                    ← FastAPI Web 服务
├── models/                     ← 预训练模型存放目录
└── docs/                       ← 部署文档
```

---

## 🚀 快速开始

### 环境要求
- #### 基础环境：Python（含 PyTorch）及 Git
<details>
<summary>详细环境需求如下</summary>

| 需求 | 推荐配置 | 说明 |
|-----|---------|------|
| **Python** | 3.10+ | 建议使用 Conda 或 venv 隔离环境 |
| **Git** | 2.0+ | 用于克隆仓库 |
| **NVIDIA GPU** | 6+ GB 显存 | YOLOv11n 仅占 ~1.5 GB；亦可使用 `--device cpu` |
| **CUDA** | 11.8+ | 需与 PyTorch 版本匹配 |
| **磁盘空间** | 15+ GB | 模型文件（~6 MB）+ 依赖包 + 视频存储 |
| **内存** | 9+ GB | 用于加载全场比赛视频帧 |
| **操作系统** | Windows 10+ / Linux | macOS 仅支持 Web/VI 分析（无 CUDA） |

> **纯 CPU 模式**: 若无可用的 GPU，使用 `--device cpu` 进行检测。推理速度约为 GPU 的 1/5 至 1/10，输出质量完全一致。VI 分析与 Web 服务无需 GPU。

</details>

- #### 模型文件：需获取 `soccana.pt`（目标检测）与 `soccana_keypoint.pt`（球场关键点检测）
> 方式一：运行 **`python scripts/download_models.py`**，脚本将从 HuggingFace（自动使用国内镜像 hf-mirror.com）下载预训练模型至 `models/` 目录。  
> 方式二：若需自行训练模型，请参考 [🔗 模型与数据集](#Models)，并将训练好的权重文件重命名后放置于 `models/` 目录下。


### 1. 本地 Web 服务

```bash
cd C:\Users\<your-name>
git clone https://github.com/han-jiahe/BlueDream_v3plus.git
cd C:\Users\<your-name>\BlueDream_v3plus\web_app
pip install -r requirements.txt
python main.py
# 浏览器打开 http://127.0.0.1:8000
# 上传视频，它将被存放于 C:\Users\<your-name>\BlueDream_v3plus\web_app\uploads
```

### 2.1. 单视频检测
建议在本地执行（Windows，GPU）

> 为避免版本冲突，建议自行构建 Python 虚拟环境（如 venv 或 conda）。
```powershell
# 激活虚拟环境后
cd C:\Users\<your-name>\BlueDream_v3plus
pip install -r requirements.txt

# 检测 + 输出 CSV + 雷达图视频
python scripts/run_detect.py --source web_app\uploads\<video> --device cuda --viz radar --video web_app\uploads\out.mp4
```

| 场景 | 建议位置 |
|------|---------|
| **单视频检测** | 本地 GPU（YOLOv11n 轻量，~1.5 GB 显存） |
| **多段批量（10+ 段）** | ModelArts Tesla T4（¥14/时），`batch_detect.py --workers 4` |

### 2.2. 视频片段的批量检测
ModelArts Notebook（租用 GPU 并发检测）
> 前提：已按 [docs/DEPLOYMENT_MODELARTS.md](docs/DEPLOYMENT_MODELARTS.md) 完成项目克隆、conda 环境配置及模型下载。以下为日常使用命令。  
> （**若本地显存 ≥ 12 GB，也可直接在本地进行批量检测**。由于华为云平台默认为中文界面，<font color="red">部署手册 [docs/DEPLOYMENT_MODELARTS.md](docs/DEPLOYMENT_MODELARTS.md) 亦为中文</font>。）

Notebook 关停后环境丢失，使用 `requirements-modelarts.txt` 可快速重建：

```bash
conda create -n soccer_v3plus python=3.10 -y && conda activate soccer_v3plus
pip install -r requirements-modelarts.txt
export HF_ENDPOINT=https://hf-mirror.com

# 将多个视频片段上传至 /path/to/videos 目录，执行批量检测
python scripts/batch_detect.py --segments_dir /path/to/videos --workers 4 --device cuda

# 若片段来自同一完整视频，合并所有 CSV
python merge_csv.py --dir /path/to/csvs --range all
```

**CSV 合并脚本参数：**

| 参数 | 必填 | 默认值 | 说明 |
|------|-----|--------|------|
| `--dir` | 是 | — | CSV 文件所在目录 |
| `--range` | 否 | `all` | 合并范围：`all`（全部） 或 `N-M`（如 `0-10`） |
| `--output` | 否 | `merged_coords.csv` | 合并输出路径 |
| `--pattern` | 否 | `coords_*.csv` | CSV 文件名匹配模式 |

### 3. 本地 Web 数据分析
返回 http://127.0.0.1:8000，上传 CSV 文件并选择分析类型。若使用了批量检测，部分分析功能存在以下限制：
- **拼接后的 CSV**：同一球员在不同片段中的 tracker ID 不一致，仅全场及球队层次的统计有效，个人层次分析不可用
- **单一片段的 CSV**：所有数据均有效，但由于时长短，全场层次的统计结果可能因样本不足而偏随机，缺乏统计意义

**<font color="red">注意：</font>** 比赛视频检测的详细流程及常见问题解答，参见 [docs/DEPLOYMENT_MODELARTS.md](docs/DEPLOYMENT_MODELARTS.md)。

---

## 📊 分析模式

系统支持**模块化分析**，可按需选择，无需一次性运行全部：

| 分析类型 | 说明 | 额外输入 |
|---------|------|---------|
| **完整分析** | $\dot{VI}$ 分布时序 + 概率密度曲线 + 球员排名 + Top 3 球员热力图 | — |
| **单球员分析** | 指定球员的 $\dot{VI}$ 时序 + 球队均值对比 + 个人热力图 | 球员 ID |
| **轨迹可视化** | 多球员在 105×68 m 标准球场上的运动轨迹（**"+足球轨迹"功能暂不支持，敬请期待**） | 球员 ID（逗号分隔） |
| **Voronoi 查看器** | 按比赛时间百分比（如 50%）查看对应帧的 Voronoi 空间剖分 | 百分比（0–100） |
| **仅 $\dot{VI}$ 分布** | $\dot{VI}$ 时序 + 概率密度 + 排名（不含热力图） | — |
| **仅热力图** | 指定球员在球场上的活动热力图 | 球员 ID |

每种分析模式独立运行，生成对应的 PNG 图表，可分别下载。

---

## 🛠️ 检测命令参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` |（必填）| 输入视频路径 |
| `--device` | `cpu` | `cuda` 或 `cpu` |
| `--output` | 自动 | CSV 输出路径 |
| `--viz` | `none` | 视频可视化：`none` / `bbox` / `radar` / `keypoints` / `all` |
| `--video` | — | 可视化视频输出（`--viz ≠ none` 时必填） |
| `--frame-count` | `-1` | 最大处理帧数（`-1` = 全部） |
| `--stride` | `1` | 帧采样间隔 |

- **`--viz` 可视化模式**：

| 值 | 输出内容 |
|----|---------|
| `none` | 仅 CSV（速度最快） |
| `bbox` | 球员边界框 + ID + 球队颜色 |
| `radar` | 战术雷达小窗叠加（右下角） |
| `keypoints` | 29 关键点 + 连接线 |
| `all` | 上述三者同时叠加 |

---

## 📋 依赖文件说明

| 文件 | 用途 |
|------|------|
| `requirements.txt` | 本地完整依赖（检测 + Web + $\dot{VI}$ 分析） |
| `requirements-modelarts.txt` | Notebook 最小依赖（仅检测，不含 Web / VI）—— 用于关停后快速重建 |
| `web_app/requirements.txt` | Web 服务专用依赖（FastAPI + 可视化） |

---

## 🔗 模型与数据集 <a id="Models"></a>

### 预训练模型

| 模型类型 | HuggingFace 仓库 | 说明 |
|---------|-----------------|------|
| **目标检测** | [Adit-jain/soccana](https://huggingface.co/Adit-jain/soccana) | YOLOv11n，检测球员、足球、裁判 |
| **关键点检测** | [Adit-jain/Soccana_Keypoint](https://huggingface.co/Adit-jain/Soccana_Keypoint) | YOLOv11n-pose，29 点球场关键点检测 |

### 训练数据集

| 数据集类型 | HuggingFace 仓库 | 说明 |
|-----------|-----------------|------|
| **关键点检测** | [Adit-jain/Soccana_Keypoint_detection_v1](https://huggingface.co/datasets/Adit-jain/Soccana_Keypoint_detection_v1) | 含 29 个球场参考点的标注数据集 |
| **目标检测** | [Adit-jain/Soccana_player_ball_detection_v1](https://huggingface.co/datasets/Adit-jain/Soccana_player_ball_detection_v1) | 球员、足球、裁判的边界框标注数据集 |

---

## 🙏 致谢

本项目基于以下开源项目及研究工作：

- **核心检测架构**: [Adit-jain/Soccer_Analysis](https://github.com/Adit-jain/Soccer_Analysis) — 完整的足球视频分析管线，提供 YOLOv11n 检测、ByteTrack 追踪、SigLIP 球队分类及 29 关键点球场注册
- **预训练模型**: [Adit-jain/soccana](https://huggingface.co/Adit-jain/soccana) 和 [Adit-jain/Soccana_Keypoint](https://huggingface.co/Adit-jain/Soccana_Keypoint) — 由 Adit Jain 训练并开源的模型权重
- **$\dot{VI}$ 分析方法**: L.R. Pereira, R.J. Lopes, J. Luka, D. Araujo, J. Ramos 等人发表于 arXiv 的信息论足球分析研究（[arXiv:2102.11182](https://doi.org/10.48550/arXiv.2102.11182)）
- **基础工具库**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)、[Supervision](https://github.com/roboflow/supervision)、[FastAPI](https://fastapi.tiangolo.com/)
