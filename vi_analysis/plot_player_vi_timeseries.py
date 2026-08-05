#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Personal VI Time Series Chart Generator
=========================================
Generates a two-panel figure for a single player:
  Top:    VI over time (line chart + rolling mean + mean/median/std lines)
  Bottom: VI density distribution (histogram + KDE, matching existing VI chart style)

Usage:
    python plot_player_vi_timeseries.py <player_id> [options]

Examples:
    # Simplest: just specify the player
    python plot_player_vi_timeseries.py 9

    # Custom input / output / fps
    python plot_player_vi_timeseries.py 9 -i my_vi_results.json -o player9.png

    # With fps to show seconds on x-axis
    python plot_player_vi_timeseries.py 9 --fps 25

    # Pick a player from team 1
    python plot_player_vi_timeseries.py 11 -t 1
"""

import argparse
import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import uniform_filter1d

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


def load_vi_data(json_path: str) -> dict:
    """Load voronoi_vi_results.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_player_vi(data: dict, player_id: int) -> tuple:
    """
    Extract frame-by-frame VI for a single player.
    Returns (frame_nums, vi_values) as numpy arrays.
    """
    frames = data.get('frames', {})
    sorted_frames = sorted(frames.keys(), key=int)
    frame_nums = []
    vi_values = []
    for fn in sorted_frames:
        fd = frames[fn]
        if str(player_id) in fd.get('player_vi', {}):
            frame_nums.append(int(fn))
            vi_values.append(fd['player_vi'][str(player_id)])
    return np.array(frame_nums), np.array(vi_values)


def plot_personal_vi_timeseries(
    json_path: str,
    player_id: int,
    team: int = 0,
    fps: float = None,
    output_path: str = 'player_vi_timeseries.png'
):
    """
    Generate the two-panel VI time series + distribution chart.

    Parameters
    ----------
    json_path  : path to voronoi_vi_results.json
    player_id  : integer player ID
    team       : team number for the title (cosmetic only)
    fps        : frames per second; if given, x-axis shows seconds instead of frame numbers
    output_path: output PNG path
    """
    data = load_vi_data(json_path)
    frame_nums, vi_vals = extract_player_vi(data, player_id)

    if len(vi_vals) == 0:
        raise ValueError(f'Player {player_id} not found in VI data. '
                         f'Check the player ID and re-run voronoi_vi.py.')

    if len(vi_vals) < 3:
        print(f'WARNING: only {len(vi_vals)} frames for Player {player_id}. '
              f'Plot may be sparse.')

    # --- Statistics ---
    mean_vi = np.mean(vi_vals)
    median_vi = np.median(vi_vals)
    std_vi = np.std(vi_vals)
    max_vi = np.max(vi_vals)
    min_vi = np.min(vi_vals)

    # --- X-axis: frames or seconds ---
    if fps and fps > 0:
        x_vals = frame_nums / fps
        x_label = 'Time (s)'
    else:
        x_vals = frame_nums
        x_label = 'Frame Number'

    # --- Colors (matching existing VI distribution chart style) ---
    line_color = '#4682B4'       # steelblue
    trend_color = '#D62728'      # red
    mean_color = '#2E8B57'       # sea green
    median_color = '#FF8C00'     # dark orange
    band_color = '#B0C4DE'       # light steelblue

    # --- Figure ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [2, 1]})

    # ===== Panel 1: VI Time Series =====
    ax_line = axes[0]

    ax_line.plot(x_vals, vi_vals, '-', color=line_color, linewidth=1.2,
                 alpha=0.7, label=f'Player {player_id} VI (raw)')
    ax_line.scatter(x_vals, vi_vals, color=line_color, s=12, alpha=0.5,
                    edgecolors='none')

    window = max(3, len(vi_vals) // 10)
    rolling = uniform_filter1d(vi_vals, size=window)
    ax_line.plot(x_vals, rolling, '-', color=trend_color, linewidth=2.5,
                 label=f'Rolling Mean (window={window})')

    ax_line.axhline(mean_vi, color=mean_color, linestyle='--', linewidth=1.5,
                    alpha=0.8, label=f'Mean = {mean_vi:.4f}')
    ax_line.axhline(median_vi, color=median_color, linestyle='--', linewidth=1.5,
                    alpha=0.8, label=f'Median = {median_vi:.4f}')

    ax_line.fill_between(x_vals, mean_vi - std_vi, mean_vi + std_vi,
                         alpha=0.12, color=band_color,
                         label=f'+/-1 Std = {std_vi:.4f}')

    ax_line.set_xlabel(x_label, fontsize=12)
    ax_line.set_ylabel('VI (Cell Area / Pitch Area)', fontsize=12)
    ax_line.set_title(
        f'Player {player_id}  |  Team {team}  |  '
        f'N={len(vi_vals)} frames  |  '
        f'Max={max_vi:.4f}  Min={min_vi:.4f}',
        fontsize=14, fontweight='bold')
    ax_line.grid(True, linestyle='--', alpha=0.5)
    ax_line.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # ===== Panel 2: VI Distribution =====
    ax_hist = axes[1]
    ax_hist.hist(vi_vals, bins=20, density=True, alpha=0.6,
                 color='steelblue', edgecolor='black', label='Histogram')

    try:
        kde = gaussian_kde(vi_vals, bw_method='scott')
        x_range = np.linspace(min(vi_vals), max(vi_vals), 200)
        kde_vals = kde(x_range)
        ax_hist.plot(x_range, kde_vals, 'r-', linewidth=2, label='KDE')
    except Exception as e:
        print(f'  [INFO] KDE skipped: {e}')

    ax_hist.axvline(mean_vi, color=mean_color, linestyle='--', linewidth=1.5,
                    label=f'Mean = {mean_vi:.4f}')
    ax_hist.axvline(median_vi, color=median_color, linestyle='--', linewidth=1.5,
                    label=f'Median = {median_vi:.4f}')

    ax_hist.set_xlabel('VI', fontsize=12)
    ax_hist.set_ylabel('Probability Density', fontsize=12)
    ax_hist.set_title('VI Distribution (Histogram + KDE)',
                      fontsize=12, fontweight='bold')
    ax_hist.grid(True, linestyle='--', alpha=0.5)
    ax_hist.legend(fontsize=9)

    plt.tight_layout(pad=3)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- Summary ---
    print(f'  Output  : {output_path}')
    print(f'  Player  : {player_id} (Team {team})')
    print(f'  Frames  : {len(vi_vals)}')
    if fps:
        print(f'  Duration: {x_vals[-1]:.1f}s (@ {fps} fps)')
    print(f'  VI Mean : {mean_vi:.6f}')
    print(f'  VI Median: {median_vi:.6f}')
    print(f'  VI Std  : {std_vi:.6f}')
    print(f'  VI Max  : {max_vi:.6f}')
    print(f'  VI Min  : {min_vi:.6f}')


def list_players(json_path: str):
    """List all player IDs found in the VI data."""
    data = load_vi_data(json_path)
    frames = data.get('frames', {})
    all_players = set()
    for fd in frames.values():
        for pid in fd.get('player_vi', {}):
            all_players.add(int(pid))
    return sorted(all_players)


# =====================================================================
# CLI
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate personal VI time series chart for a single player.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_player_vi_timeseries.py 9
  python plot_player_vi_timeseries.py 9 --team 0 --fps 25 -o out.png
  python plot_player_vi_timeseries.py 11 -i my_vi.json --list
        ''')

    parser.add_argument('player_id', type=int, nargs='?', default=None,
                        help='Player ID to plot (omit with --list to see all players)')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Path to voronoi_vi_results.json (default: <script_dir>/voronoi_vi_results.json)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output PNG path (default: <script_dir>/football_analysis_output/data/player<id>_vi_timeseries.png)')
    parser.add_argument('-t', '--team', type=int, default=None,
                        help='Team number for title (default: auto-detected or 0)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Frames per second; if given, x-axis shows seconds instead of frame numbers')
    parser.add_argument('--list', action='store_true',
                        help='List all player IDs found in the VI data, then exit')

    args = parser.parse_args()

    # ---- Resolve paths ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vi_json = args.input or os.path.join(script_dir, 'voronoi_vi_results.json')

    if not os.path.exists(vi_json):
        print(f'ERROR: VI data file not found: {vi_json}')
        print(f'  Run voronoi_vi.py first to generate this file.')
        sys.exit(1)

    # ---- List mode ----
    if args.list or args.player_id is None:
        players = list_players(vi_json)
        print(f'Players found in {vi_json}: {len(players)}')
        for pid in players:
            print(f'  Player {pid}')
        if not args.list:
            print('\nUsage: python plot_player_vi_timeseries.py <player_id> [options]')
        sys.exit(0)

    # ---- Resolve team (try to read from personal_vi.json if not given) ----
    team = args.team
    if team is None:
        # Try autodetect from personal_vi.json (same output dir)
        out_dir = os.path.join(script_dir, 'football_analysis_output', 'data')
        pvj = os.path.join(out_dir, 'personal_vi.json')
        if os.path.exists(pvj):
            with open(pvj, 'r', encoding='utf-8') as f:
                pdata = json.load(f)
            for p in pdata.get('players_ranked', []):
                if p['player_id'] == args.player_id:
                    team = p['team']
                    break
        if team is None:
            team = 0

    # ---- Resolve output path ----
    out_dir = os.path.join(script_dir, 'football_analysis_output', 'data')
    output = args.output or os.path.join(out_dir, f'player{args.player_id}_vi_timeseries.png')

    # ---- Generate ----
    try:
        plot_personal_vi_timeseries(
            json_path=vi_json,
            player_id=args.player_id,
            team=team,
            fps=args.fps,
            output_path=output
        )
        print('Done.')
    except ValueError as e:
        print(f'ERROR: {e}')
        print(f'  Run with --list to see available player IDs.')
        sys.exit(1)
