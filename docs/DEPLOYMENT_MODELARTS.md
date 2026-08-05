# BlueDream v3plus — ModelArts Notebook 部署指南

> 基于华为云的**足球比赛视频视觉识别**的云端环境搭建  
> ☁️ GPU 推理：ModelArts T4 | 本地 Web：Windows FastAPI 
> （本方案仅介绍华为云的部署指南，有其他GPU环境租用T4显卡的方法也类似）

---

## 1. 架构概览

```
本地 Windows (Web GUI)              ModelArts Notebook (GPU)
┌────────────────────────┐            ┌──────────────────────┐
│  python main.py        │            │  python scripts/     │
│  http://127.0.0.1:8000 │ ───上传──▶      run_detect.py     │
│                        │ JupyterLab │    --device cuda     │
│  视频上传 → CSV 下载    │  文件传输   │  batch_detect.py     │
│  上传 → VI 分析 → 图表  │            │                      │
└────────────────────────┘            └──────────────────────┘
```

**工作流**：本地 Web 上传视频 → 传到 Notebook → GPU 检测 → 下载 CSV → 本地 $\dot{VI}$ 分析 → 图表

---

## 2. 创建 ModelArts Notebook

### 2.1 配置

| 选项 | 值 |
|------|-----|
| 镜像(预置) | `PyTorch-2.1.0-cuda12.1-py3.10.6-ubuntu22.04-x86_64` |
| 实例规格(公共)  | GPU / Tesla T4 16GB |
| 存储(磁盘) | 30GB |
| 自动停止 | 1h 或 2h |
> 其他选项默认即可

### 2.2 文件传输

#### 推荐方式 — JupyterLab 网页上传：
1. 点击"打开"进入 JupyterLab
2. 使用文件浏览器中的上传按钮
3. 进入目标目录，上传视频文件
4. 下载结果：右键文件 → "Download"

> 适用于 500MB 以内的文件。大文件建议通过 obsutil 经 OBS 中转，具体方法不赘述。

#### 备选方式 — SCP 命令行传输（需 Notebook 支持公网 SSH）：

> 适用场景：文件较大、批量下载、脚本自动化。  
> 前置要求：
> 1）进入 "密码安全中心DEW / 密钥对管理-账号密钥对 / 创建账号密钥对" 中，勾选 *密钥对管理服务免责声明* 后创建 SSH 密钥对并下载 `.pem` 私钥文件到本地 `C:\Users\{{user}}\%USERPROFILE%\.ssh\` 目录；
> 2）关停notebook实例，进入 "基本信息" 中打开 "SSH远程开发" 并选择刚才创建的密钥对即可。  
> 完成以上步骤后也可在本地VS Code中直接使用notebook，详情参考[在VS Code中一键连接Notebook](https://support.huaweicloud.com/devtool-modelarts/devtool-modelarts_0206.html)，以下为手动传输文件方法。

- **Step 1 — 获取 Notebook 的 SSH 连接信息**

在 "Notebook / 远程访问" 标签中，可看到类似以下格式的 SSH 地址：

```
ssh://ma-user@dev-modelarts-cnnorth4.huaweicloud.com:31248
```

各字段含义如下：

| 字段 | 含义 | 示例值 |
|------|------|--------|
| `ma-user` | 默认登录用户名（固定） | `ma-user` |
| `dev-modelarts-cnnorth4.huaweicloud.com` | Notebook 的公网域名 | 每台实例不同 |
| `31248` | SSH 端口（**ModelArts 动态分配的高位端口，非默认 22**） | 每台实例不同 |

> ⚠️ 以上三个值均需从你的 Notebook 详情页获取，不能直接复制示例值。端口号尤其关键——它是 ModelArts 为安全原因分配的随机高位端口，每次创建新实例都会变化。

- **Step 2 — 执行 SCP 传输**

将你在详情页看到的域名和端口替换到以下命令中：

```powershell
# 在本地C:\Users\{{user}}目录中上传视频到 Notebook（将 <host> 和 <port> 替换为实际值）
scp -i %USERPROFILE%\.ssh\notebook-key.pem -P <port> video.mp4 ma-user@<host>:/home/ma-user/work/

# 从 Notebook 下载 CSV 到本地
scp -i %USERPROFILE%\.ssh\notebook-key.pem -P <port> ma-user@<host>:/home/ma-user/work/BlueDream_v3plus/coords_*.csv .

# 示例（假设 Notebook 地址为 dev-modelarts-cnnorth4.huaweicloud.com，端口 31248）：
scp -i %USERPROFILE%\.ssh\notebook-key.pem -P 31248 video.mp4 ma-user@dev-modelarts-cnnorth4.huaweicloud.com:/home/ma-user/work/
```

---

## 3. 环境配置（简化版 — 现代 PyTorch）

> **v3 的重大改进**：不再锁定 torch 1.12.1 CUDA 11.3。
> 我们使用现代 PyTorch 2.x + CUDA 12.1，与 ModelArts T4 完全兼容。

```bash
# 1. 克隆项目
cd /home/ma-user/work
git clone https://github.com/han-jiahe/BlueDream_v3plus.git
cd BlueDream_v3plus

# 2. 创建 conda 环境
conda create -n soccer_v3plus python=3.10 -y
conda activate soccer_v3plus

# 3. 安装依赖（简洁 — 不再需要锁定版本！）
pip install "numpy<2"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics>=8.2.0 supervision>=0.22.0 opencv-python-headless
pip install scikit-learn>=1.3.0

# 4. 验证 GPU
python -c "import torch; t=torch.zeros(1).cuda(); print('CUDA OK:', torch.cuda.get_device_name(0))"
# 必须输出: CUDA OK: Tesla T4

# 5. 设置 HuggingFace 镜像（国内加速）
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# 6. 下载模型
python scripts/download_models.py
```

**注意**：soccana 模型首次使用时会从 HuggingFace 自动下载。关键点模型需要手动放置或从 OBS 下载。

---

## 4. GPU 检测策略

YOLOv11n (nano) 是轻量模型（2.6M 参数），仅占用 ~1.5GB 显存。瓶颈在于 I/O 而非 GPU 计算——单视频检测速度受限于帧读取 + SigLIP 嵌入提取 + 串行处理，而非 GPU 算力。因此：

| 场景 | 运行位置 | 原因 |
|------|---------|------|
| **单视频检测** | **本地 GPU**（自己的电脑） | ModelArts T4 不会更快；无延迟收益 |
| **多段批量（10+ 个视频）** | **ModelArts T4** | 通过 `batch_detect.py --workers 4~5` 实现多 GPU 并发 |

### 4.1 单视频检测（本地）

```bash
cd C:\Users\<your-name>\BlueDream_v3plus
python scripts/run_detect.py \
  --source video.mp4 \
  --device cuda \
  --viz radar \
  --video output_annotated.mp4
```

**输出：**
- `coords_<video_name>.csv` — 球员追踪数据（v3 兼容格式）
- `output_annotated.mp4` — 带可视化叠加的标注视频

**参数说明：**

| 参数 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `--source` | 是 | — | 输入视频路径 |
| `--device` | 否 | `cpu` | `cuda` 或 `cpu` |
| `--output` | 否 | 自动 | CSV 输出路径（默认：`<视频目录>/coords_<视频名>.csv`） |
| `--viz` | 否 | `none` | 视频可视化模式：`none` / `bbox` / `radar` / `keypoints` / `all` |
| `--video` | 当 `--viz ≠ none` 时必填 | — | 可视化视频输出路径 |
| `--frame-count` | 否 | `-1`（全部） | 最大处理帧数 |
| `--stride` | 否 | `1` | 帧间隔 |

**`--viz` 可视化模式：**

| 值 | 输出内容 |
|----|---------|
| `none` | 仅 CSV（最快，不编码视频） |
| `bbox` | 球员边界框 + ID + 球队颜色 |
| `radar` | 战术雷达小窗叠加（右下角） |
| `keypoints` | 29 球场关键点 + 连线 |
| `all` | 三者全部叠加 |

### 4.2 批量检测 — ModelArts T4（多段并发）

仅当你需要同时处理 **多个的视频片段** 时才使用 ModelArts。

> **价格**：Tesla T4 — **¥14/小时**（华为云 ModelArts，截至 2026 年）

```bash
# 在 ModelArts Notebook 上（完成第 3 节的环境配置后）
conda activate soccer_v3plus
export HF_ENDPOINT=https://hf-mirror.com

python scripts/batch_detect.py \
  --segments_dir /path/to/videos \
  --workers 5 \
  --device cuda
```

| 参数 | 说明 |
|------|------|
| `--segments_dir` | 视频文件目录 |
| `--workers` | 并发数（T4 16GB → 2-3；更多 worker ≠ 更快，受 I/O 限制） |
| `--device` | `cuda` 或 `cpu` |
| `--output_dir` | CSV 输出目录（默认：同 segments_dir） |
| `--pattern` | 视频文件名匹配模式（默认：`segment_*.mp4`） |

---

## 5. 本地 Web 服务

### 5.1 安装依赖

```powershell
cd C:\Users\<your-name>
git clone https://github.com/han-jiahe/BlueDream_v3plus.git
cd C:\Users\<your-name>\BlueDream_v3plus\web_app
pip install -r requirements.txt
```

### 5.2 启动

```bash
python main.py
```

浏览器自动打开 `http://127.0.0.1:8000`

### 5.3 使用流程

| 步骤 | 操作 |
|------|------|
| 1 | 上传视频至 Web（本地检测单个视频时） |
| 1 | 上传视频文件夹至 Notebook（云端检测批量视频时）|
| 2 | 运行相应的检测命令 |
| 3 | 下载生成的 CSV ，对于批量检测可以考虑`python merge_csv.py --dir /path/to/csvs`合并所有 CSV|
| 4 | 上传 CSV 到本地 Web |
| 5 | 选择分析类型，运行分析 |
| 6 | 查看输出图表 |

---

## 6. 分析类型

| 类型 | 生成的图表 | 额外参数 |
|------|-----------|---------|
| **完整分析** | $\dot{VI}$ 分布时序 + 概率密度 + 球员排名 + Top3 热力图 | — |
| **单球员分析** | $\dot{VI}$ 时序 + 球队对比 + 热力图 | 球员 ID |
| **轨迹可视化** | 105×68m 球场的多球员轨迹（**“同时显示足球轨迹”** 功能暂不支持，敬请期待） | 球员 ID（逗号分隔） |
| **Voronoi 查看器** | 按比赛百分比显示 Voronoi 剖分 | 百分比（0-100） |
| **仅 $\dot{VI}$ 分布** | $\dot{VI}$ 时序 + 概率密度 + 球员排名 | — |
| **仅热力图** | 指定球员热力图 + colorbar | 球员 ID |

---

## 7. 输出格式

CSV（v3 兼容）：`帧号, 球员ID, 画面X, 画面Y, 雷达X, 雷达Y, 真实X(米), 真实Y(米), 队伍`

- `队伍`: 0 = 主队, 1 = 客队
- 坐标原点：`真实X(米)` 向右，`真实Y(米)` 向下（图像约定）；VI 分析时自动翻转为左下角原点

---

## 8. Notebook 关停后的重新激活

华为云 ModelArts Notebook 经过你设置的闲置时长后会自动停止。**只有 `/home/ma-user/work/` 目录被保留**——conda 环境和其他系统文件会在重启时丢失。每次都需要从头重建环境。

### 8.1 完整重建（重启后）

```bash
cd /home/ma-user/work/BlueDream_v3plus

conda create -n soccer_v3plus python=3.10 -y
conda activate soccer_v3plus
```
```bash
pip install "numpy<2"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics>=8.2.0 supervision>=0.22.0 opencv-python-headless
pip install scikit-learn>=1.3.0
```
```bash
export HF_ENDPOINT=https://hf-mirror.com

python -c "import torch; t=torch.zeros(1).cuda(); print('CUDA OK:', torch.cuda.get_device_name(0))"
```

### 8.2 运行检测

```bash
conda activate soccer_v3plus
cd /home/ma-user/work/BlueDream_v3plus
export HF_ENDPOINT=https://hf-mirror.com

# 单视频
python scripts/run_detect.py --source <video> --device cuda

# 批量
python scripts/batch_detect.py --segments_dir /path/to/videos --workers 5 --device cuda
```

> **提示**：模型文件（`models/*.pt`）在 `work/` 目录下，重启后不会丢失，无需重新下载。

---

## 9. 性能说明

| 优化项 | 效果 |
|--------|------|
| YOLOv11n (nano) | 2.6M 参数，T4 上 30+ FPS |
| FP16 推理 | GPU 2× 加速（Tensor Cores） |
| 调节 `--frame-count 15` 使每 15 帧检测一次球场 | 跳过多余的关键点检测 |
| 批量模式（仅 CSV） | 跳过视频编码，节省 ~30% 时间 |
| 4-5 个并发 worker | 充分利用 GPU 显存 |

---

## 10. 与 v3 (roboflow/sports) 的关键区别

| 方面 | v3 (roboflow/sports) | v3plus (soccana) |
|------|---------------------|-------------------|
| 检测模型 | 3 个独立 YOLO 模型 | 1 个统一 YOLOv11n |
| 检测类别 | Ball, GK, Player, Ref (4 类) | Player, Ball, Referee (3 类) |
| 球队分类 | SigLIP + UMAP + KMeans | SigLIP + UMAP + KMeans |
| Torch 版本 | 1.12.1+cu113（锁定） | 2.x（灵活） |
| 依赖 | roboflow/sports, transformers | ultralytics + supervision + transformers |
| 安装复杂度 | 高（6 步顺序安装） | 低（3 步 pip install） |
