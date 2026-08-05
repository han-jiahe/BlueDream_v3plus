"""
BlueDream v3plus — FastAPI Web Application
Video upload → ModelArts GPU detection → CSV download → Local VI analysis
"""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import BASE_DIR, UPLOAD_DIR, OUTPUT_DIR, ALLOWED_EXTENSIONS
from pipeline import (
    merge_csvs,
    compute_voronoi_vi, compute_vi_distribution,
    generate_vi_chart, generate_player_ranking, generate_heatmap,
    generate_player_timeseries, generate_vi_density,
    generate_player_with_team_vi, generate_voronoi_frame,
    generate_multi_player_trajectory, generate_ball_trajectory,
    _load_team_map,
    csv_to_samples,
)


# ── Analysis Types ────────────────────────────────────

ANALYSIS_TYPES = {
    "full": "Full Analysis (VI Distribution + Ranking + Density + Top3 Heatmaps)",
    "player": "Single Player Analysis (VI + Team Comparison + Heatmap)",
    "trajectory": "Trajectory Visualization (Multi-player + Optional Ball)",
    "voronoi": "Voronoi Viewer (select by match %)",
    "vi_only": "VI Distribution Only (Time-series + Density + Ranking)",
    "heatmap": "Heatmap Only",
}

app = FastAPI(title="BlueDream v3plus")

# Templates — inline simple HTML (no Jinja2 dependency)
templates_dir = BASE_DIR / "templates"
templates_dir.mkdir(exist_ok=True)

# Static files
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/results-files", StaticFiles(directory=str(OUTPUT_DIR)), name="results_files")


# ── Routes ────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Main upload page."""
    upload_html = templates_dir / "upload.html"
    if upload_html.exists():
        return FileResponse(str(upload_html))
    return HTMLResponse(_DEFAULT_UPLOAD_HTML)


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload full video (no splitting — preserves player ID consistency)."""
    if not file.filename:
        return JSONResponse({"error": "No file"}, status_code=400)

    ext = Path(file.filename).suffix
    if ext.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ts"}:
        return JSONResponse({"error": f"Unsupported format: {ext}"}, status_code=400)

    video_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{video_id}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f, length=8 * 1024 * 1024)

    return JSONResponse({
        "video_id": video_id,
        "original_name": file.filename,
        "size_mb": round(save_path.stat().st_size / 1e6, 1),
        "hint": "Transfer video to ModelArts Notebook, run GPU detection, upload resulting CSV back.",
    })


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), video_id: str = Form("")):
    """Upload ModelArts detection result CSV."""
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    task_dir = OUTPUT_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    save_path = task_dir / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f, length=8 * 1024 * 1024)

    csv_count = len(list(task_dir.glob("*.csv")))

    return JSONResponse({
        "task_id": task_id,
        "csv_file": file.filename,
        "csv_count": csv_count,
    })


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    video_id: str = Form(""),
    task_id: str = Form(""),
    analysis_type: str = Form("full"),
    player_id: str = Form(""),
    include_ball: str = Form("0"),
):
    """Run VI analysis on detection CSV."""
    if task_id:
        task_dir = OUTPUT_DIR / task_id
    elif video_id:
        task_dir = OUTPUT_DIR / video_id
    else:
        return HTMLResponse("<h3>Missing parameters</h3>", status_code=400)

    if not task_dir.exists():
        return HTMLResponse(f"<h3>Task directory not found: {task_dir}</h3>", status_code=404)

    csv_files = sorted(task_dir.glob("*.csv"))
    if not csv_files:
        return HTMLResponse("<h3>No CSV files found</h3>", status_code=400)

    merged_csv = task_dir / "merged_coords.csv"
    if len(csv_files) > 1:
        merge_csvs(csv_files, merged_csv)
    else:
        merged_csv = csv_files[0]

    try:
        charts = []

        # Voronoi VI (needed for most analysis types)
        voronoi_json = compute_voronoi_vi(merged_csv)

        # Clustering VI
        samples_json = task_dir / "samples.json"
        vi_json = None
        try:
            csv_to_samples(merged_csv, samples_json)
            vi_json = compute_vi_distribution(samples_json)
        except Exception as e:
            print(f"[WARN] Clustering VI failed (non-fatal): {e}")

        team_map = _load_team_map(merged_csv)

        # ── Generate charts by analysis type ──
        if analysis_type == "full":
            if vi_json:
                charts.append(("VI Distribution Timeline", generate_vi_chart(vi_json).name))
                charts.append(("VI Probability Density", generate_vi_density(vi_json).name))
            rc = generate_player_ranking(voronoi_json)
            if rc:
                charts.append(("Player VI Ranking", rc.name))
            # Top 3 players heatmaps
            with open(voronoi_json, "r", encoding="utf-8") as f:
                vdata = json.load(f)
            player_total_vi = {}
            for fdata in vdata.get("frames", {}).values():
                for pid, vi_val in fdata.get("player_vi", {}).items():
                    player_total_vi[pid] = player_total_vi.get(pid, 0.0) + float(vi_val)
            top_players = sorted(player_total_vi, key=player_total_vi.get, reverse=True)[:3]
            for pid_str in top_players:
                pid = int(pid_str)
                team_tag = f"({team_map.get(pid, '?')})" if pid in team_map else ""
                charts.append(
                    (f"Player {pid} {team_tag} Heatmap (Total VI: {player_total_vi[pid_str]:.3f})",
                     generate_heatmap(merged_csv, pid).name)
                )

        elif analysis_type == "player" and player_id:
            pid = int(player_id.strip())
            team_tag = f"({team_map.get(pid, '?')})" if pid in team_map else ""
            charts.append(
                (f"Player {pid} {team_tag} vs Team VI",
                 generate_player_with_team_vi(voronoi_json, merged_csv, pid).name)
            )
            charts.append(
                (f"Player {pid} {team_tag} Heatmap",
                 generate_heatmap(merged_csv, pid).name)
            )

        elif analysis_type == "vi_only":
            if vi_json:
                charts.append(("VI Distribution Timeline", generate_vi_chart(vi_json).name))
                charts.append(("VI Probability Density", generate_vi_density(vi_json).name))
            rc = generate_player_ranking(voronoi_json)
            if rc:
                charts.append(("Player VI Ranking", rc.name))

        elif analysis_type == "voronoi":
            pct = float(player_id) if player_id else 50
            charts.append(
                (f"Voronoi @ {pct:.0f}%",
                 generate_voronoi_frame(voronoi_json, merged_csv, pct).name)
            )

        elif analysis_type == "trajectory" and player_id:
            ids = [int(x.strip()) for x in player_id.replace(",", " ").split() if x.strip()]
            show_ball = include_ball == "1"
            if ids:
                traj_path = generate_multi_player_trajectory(
                    merged_csv, ids, task_dir.name, include_ball=show_ball
                )
                charts.append(("Player Trajectories", traj_path.name))
            if show_ball:
                ball_path = generate_ball_trajectory(merged_csv, task_dir.name)
                if ball_path:
                    charts.append(("Ball Trajectory", ball_path.name))

        elif analysis_type == "heatmap" and player_id:
            pid = int(player_id.strip())
            charts.append((f"Player {pid} Heatmap", generate_heatmap(merged_csv, pid).name))

        else:
            # Default: same as full
            if vi_json:
                charts.append(("VI Distribution Timeline", generate_vi_chart(vi_json).name))
            rc = generate_player_ranking(voronoi_json)
            if rc:
                charts.append(("Player VI Ranking", rc.name))

        # ── Build results HTML ──
        task_name = task_dir.name
        n_files = len(csv_files)
        chart_html = "".join(
            f'<div class="card"><h3>{label}</h3>'
            f'<img src="/results-files/{task_name}/{path}" class="chart-img"></div>'
            for label, path in charts
        )
        html = (
            '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Analysis Results</title>'
            '<link rel="stylesheet" href="/static/style.css"></head><body>'
            '<main class="container"><h2>Analysis Results</h2>'
            f'<p>Task: <code>{task_name}</code> | CSV files: {n_files}</p>'
            f'{chart_html}'
            '<a href="/" class="btn" style="margin-top:2rem">Back</a>'
            '</main></body></html>'
        )
        return HTMLResponse(content=html, status_code=200)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(
            f"<h3>Analysis Failed</h3><pre>{traceback.format_exc()}</pre>", status_code=500,
        )


@app.get("/results/{task_id}")
async def view_results(request: Request, task_id: str):
    """View existing analysis results."""
    task_dir = OUTPUT_DIR / task_id
    if not task_dir.exists():
        return HTMLResponse("<h3>Task not found</h3>", status_code=404)

    pngs = sorted(task_dir.glob("*.png"))
    csvs = sorted(task_dir.glob("*.csv"))
    jsons = sorted(task_dir.glob("*.json"))

    chart_html = "".join(
        f'<div class="card"><h3>Chart</h3>'
        f'<img src="/results-files/{task_id}/{p.name}" class="chart-img"></div>'
        for p in pngs
    )
    file_html = "".join(
        f'<li><a href="/results-files/{task_id}/{f.name}" download>{f.name}</a></li>'
        for f in list(csvs) + list(jsons)
    )
    html = (
        '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Results</title>'
        '<link rel="stylesheet" href="/static/style.css"></head><body>'
        '<main class="container">'
        f'<h2>Task: {task_id}</h2>'
        f'{chart_html}'
        f'<div class="card"><h3>Files</h3><ul>{file_html}</ul></div>'
        '<a href="/" class="btn">Back</a>'
        '</main></body></html>'
    )
    return HTMLResponse(content=html, status_code=200)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": "BlueDream v3plus",
        "model": "YOLOv11n soccana",
    }


# ── Default Upload HTML (fallback if no template file) ──

_DEFAULT_UPLOAD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BlueDream v3plus</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1923; color: #e0e0e0; min-height: 100vh; }
  .container { max-width: 900px; margin: 0 auto; padding: 2rem; }
  h1 { color: #FF1493; margin-bottom: 0.5rem; }
  .subtitle { color: #888; margin-bottom: 2rem; font-size: 0.9rem; }
  .card { background: #1a2732; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
  .card h3 { color: #00BFFF; margin-bottom: 1rem; }
  input, select, button { padding: 10px 16px; border-radius: 8px; border: 1px solid #333;
    background: #0f1923; color: #e0e0e0; font-size: 14px; width: 100%; margin-bottom: 0.8rem; }
  button { background: #FF1493; color: white; border: none; cursor: pointer; font-weight: 600; }
  button:hover { background: #ff3399; }
  .btn { display: inline-block; padding: 10px 24px; background: #FF1493; color: white;
    text-decoration: none; border-radius: 8px; font-weight: 600; }
  .btn:hover { background: #ff3399; }
  .chart-img { max-width: 100%; border-radius: 8px; margin: 0.5rem 0; }
  ul { padding-left: 1.5rem; }
  pre { background: #111; padding: 1rem; border-radius: 8px; overflow-x: auto; }
</style>
</head>
<body>
<main class="container">
  <h1>BlueDream v3plus</h1>
  <p class="subtitle">Football Video Analysis — YOLOv11n soccana + Voronoi VI</p>

  <div class="card">
    <h3>1. Upload Video</h3>
    <form id="upload-video-form">
      <input type="file" name="file" accept="video/*" required>
      <button type="submit">Upload Video</button>
    </form>
    <p id="video-status" style="margin-top:0.5rem; color:#00BFFF;"></p>
  </div>

  <div class="card">
    <h3>2. Upload Detection CSV (from ModelArts GPU)</h3>
    <form id="upload-csv-form">
      <input type="file" name="file" accept=".csv" required>
      <input type="text" name="video_id" id="csv-video-id" placeholder="Video ID (optional)">
      <button type="submit">Upload CSV</button>
    </form>
    <p id="csv-status" style="margin-top:0.5rem; color:#00BFFF;"></p>
  </div>

  <div class="card">
    <h3>3. Run Analysis</h3>
    <form action="/analyze" method="post" enctype="multipart/form-data">
      <input type="text" name="task_id" id="analyze-task-id"
             placeholder="Task ID (from CSV upload)" required>
      <select name="analysis_type">
        <option value="full">Full Analysis</option>
        <option value="player">Single Player</option>
        <option value="trajectory">Trajectory</option>
        <option value="voronoi">Voronoi Viewer</option>
        <option value="vi_only">VI Distribution Only</option>
        <option value="heatmap">Heatmap Only</option>
      </select>
      <input type="text" name="player_id" placeholder="Player ID (for player/heatmap/trajectory)">
      <label style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.8rem;">
        <input type="checkbox" name="include_ball" value="1" style="width:auto;">
        Include Ball Trajectory
      </label>
      <button type="submit">Run Analysis</button>
    </form>
  </div>
</main>

<script>
document.getElementById('upload-video-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const fd = new FormData(form);
  const st = document.getElementById('video-status');
  st.textContent = 'Uploading...';
  const res = await fetch('/upload-video', { method: 'POST', body: fd });
  const data = await res.json();
  if (data.video_id) {
    st.textContent = `Uploaded: ${data.video_id} (${data.size_mb} MB)`;
    document.getElementById('csv-video-id').value = data.video_id;
  } else {
    st.textContent = 'Error: ' + JSON.stringify(data);
  }
});

document.getElementById('upload-csv-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const fd = new FormData(form);
  const st = document.getElementById('csv-status');
  st.textContent = 'Uploading...';
  const res = await fetch('/upload-csv', { method: 'POST', body: fd });
  const data = await res.json();
  if (data.task_id) {
    st.textContent = `Task: ${data.task_id} (${data.csv_count} CSV files)`;
    document.getElementById('analyze-task-id').value = data.task_id;
  } else {
    st.textContent = 'Error: ' + JSON.stringify(data);
  }
});
</script>
</body>
</html>"""
