# -*- coding: utf-8 -*-
"""
Visualization for Voronoi VI analysis.
Draws pitch + Voronoi cells + player dots + VI bar charts.
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MPLPolygon
from collections import defaultdict

# Add parent for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voronoi_vi import (
    compute_voronoi_vi_from_csv, compute_frame_vi,
    PITCH_LENGTH, PITCH_WIDTH, PITCH_AREA
)


# ============================================================
# Color scheme
# ============================================================
TEAM_COLORS = {
    0: '#e74c3c',   # red
    1: '#3498db',   # blue
}
TEAM_NAMES = {0: 'Team 0', 1: 'Team 1'}
PITCH_GREEN = '#2ecc71'
PITCH_LINE = '#ffffff'


def draw_pitch_lines(ax):
    """Draw soccer pitch outline and markings on a matplotlib Axes."""
    L, W = PITCH_LENGTH, PITCH_WIDTH

    # Pitch outline
    ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], color=PITCH_LINE, linewidth=2)

    # Center line
    ax.plot([L/2, L/2], [0, W], color=PITCH_LINE, linewidth=2)

    # Center circle (radius 9.15m)
    center_circle = plt.Circle((L/2, W/2), 9.15, fill=False,
                                color=PITCH_LINE, linewidth=2)
    ax.add_patch(center_circle)

    # Center spot
    ax.plot(L/2, W/2, 'o', color=PITCH_LINE, markersize=4)

    # Penalty areas (16.5m x 40.32m -> 16.5 x 40.3)
    # Left penalty area
    ax.plot([0, 16.5, 16.5, 0],
            [(W-40.32)/2, (W-40.32)/2, (W+40.32)/2, (W+40.32)/2],
            color=PITCH_LINE, linewidth=2)
    # Right penalty area
    ax.plot([L, L-16.5, L-16.5, L],
            [(W-40.32)/2, (W-40.32)/2, (W+40.32)/2, (W+40.32)/2],
            color=PITCH_LINE, linewidth=2)

    # Goal areas (5.5m x 18.32m)
    ax.plot([0, 5.5, 5.5, 0],
            [(W-18.32)/2, (W-18.32)/2, (W+18.32)/2, (W+18.32)/2],
            color=PITCH_LINE, linewidth=2)
    ax.plot([L, L-5.5, L-5.5, L],
            [(W-18.32)/2, (W-18.32)/2, (W+18.32)/2, (W+18.32)/2],
            color=PITCH_LINE, linewidth=2)

    # Penalty spots
    ax.plot(11.0, W/2, 'o', color=PITCH_LINE, markersize=4)
    ax.plot(L-11.0, W/2, 'o', color=PITCH_LINE, markersize=4)

    # Corner arcs (simplified as small arcs)
    for cx, cy in [(0,0), (0,W), (L,0), (L,W)]:
        corner = mpatches.Arc((cx, cy), 2, 2, angle=0, theta1=0, theta2=90,
                               color=PITCH_LINE, linewidth=1.5)
        ax.add_patch(corner)


def draw_voronoi_frame(results, frame_num, ax, show_labels=False):
    """
    Draw Voronoi cells and player positions for a single frame.

    Args:
        results: dict from compute_voronoi_vi_from_csv
        frame_num: frame number to visualize
        ax: matplotlib Axes
        show_labels: whether to show player ID labels
    """
    if frame_num not in results:
        print(f'Frame {frame_num} not found in results.')
        return

    frame_data = results[frame_num]
    cell_polygons = frame_data.get('cell_polygons', {})
    player_vi = frame_data.get('player_vi', {})

    # We need player coordinates — re-read from CSV to get team info
    # Actually, we need to store team info in results. Let's handle this.
    # For now, re-read from CSV.
    pass


def visualize_frame(
    csv_path: str,
    frame_num: int,
    output_path: str = None,
    show_bar_chart: bool = True
):
    """
    Generate a complete Voronoi VI visualization for one frame.

    Layout:
      [  Pitch + Voronoi Cells  |  Individual VI Bar Chart  ]

    Args:
        csv_path: Path to the player coordinates CSV
        frame_num: Frame number to visualize
        output_path: If provided, save figure to this path
        show_bar_chart: Whether to include the bar chart panel
    """
    # Read players for this frame
    players = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['帧号']) == frame_num and int(row['队伍']) != 3:
                players.append({
                    'player_id': int(row['球员ID']),
                    'x': float(row['真实X(米)']),
                    'y': float(row['真实Y(米)']),
                    'team': int(row['队伍'])
                })

    if not players:
        print(f'No players found for frame {frame_num}.')
        return

    # Compute VI for this frame
    frame_data = compute_frame_vi(players)

    # Build figure
    if show_bar_chart:
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.05)
        ax_pitch = fig.add_subplot(gs[0, 0])
        ax_bar = fig.add_subplot(gs[0, 1])
    else:
        fig, ax_pitch = plt.subplots(1, 1, figsize=(12, 8))
        ax_bar = None

    # ---- Draw pitch ----
    ax_pitch.set_facecolor(PITCH_GREEN)
    ax_pitch.set_xlim(-2, PITCH_LENGTH + 2)
    ax_pitch.set_ylim(-2, PITCH_WIDTH + 2)
    ax_pitch.set_aspect('equal')
    ax_pitch.axis('off')
    draw_pitch_lines(ax_pitch)

    # ---- Draw Voronoi cells ----
    cell_polygons = frame_data['cell_polygons']

    # Build team lookup
    player_team = {p['player_id']: p['team'] for p in players}

    for pid, poly in cell_polygons.items():
        if poly is None or poly.is_empty:
            continue
        team = player_team.get(pid, -1)
        color = TEAM_COLORS.get(team, '#95a5a6')

        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax_pitch.fill(x, y, alpha=0.35, color=color, edgecolor='white',
                          linewidth=0.5)
        elif poly.geom_type == 'MultiPolygon':
            for part in poly.geoms:
                x, y = part.exterior.xy
                ax_pitch.fill(x, y, alpha=0.35, color=color, edgecolor='white',
                              linewidth=0.5)

    # ---- Draw player positions ----
    for p in players:
        color = TEAM_COLORS.get(p['team'], 'gray')
        ax_pitch.scatter(p['x'], p['y'], c=color, s=80, edgecolors='white',
                         linewidths=1.5, zorder=5)
        ax_pitch.annotate(str(p['player_id']), (p['x'], p['y']),
                          textcoords='offset points', xytext=(0, 8),
                          fontsize=7, color='white', ha='center',
                          fontweight='bold', zorder=6,
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                                     alpha=0.6))

    # ---- Legend ----
    legend_elements = [
        mpatches.Patch(facecolor=TEAM_COLORS[0], alpha=0.35, label=TEAM_NAMES[0]),
        mpatches.Patch(facecolor=TEAM_COLORS[1], alpha=0.35, label=TEAM_NAMES[1]),
    ]
    ax_pitch.legend(handles=legend_elements, loc='lower left',
                    fontsize=9, framealpha=0.8)

    # ---- Title on pitch ----
    team_vi = frame_data['team_vi']
    title = f'Frame {frame_num}  |  '
    title += f'{TEAM_NAMES[0]} VI: {team_vi.get(0,0):.4f}  |  '
    title += f'{TEAM_NAMES[1]} VI: {team_vi.get(1,0):.4f}'
    ax_pitch.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # ---- Bar chart (individual VI) ----
    if ax_bar is not None:
        player_vi = frame_data['player_vi']

        # Sort by VI descending
        sorted_vi = sorted(player_vi.items(), key=lambda x: -x[1])
        pids = [str(pid) for pid, _ in sorted_vi]
        vis = [v for _, v in sorted_vi]
        teams_in_order = [player_team.get(int(pid), -1) for pid, _ in sorted_vi]
        colors_in_order = [TEAM_COLORS.get(t, '#95a5a6') for t in teams_in_order]

        y_pos = range(len(pids))
        ax_bar.barh(y_pos, vis, color=colors_in_order, edgecolor='white',
                     linewidth=0.8, alpha=0.85)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(pids, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel('Individual VI', fontsize=10)
        ax_bar.set_title('Individual Voronoi Influence', fontsize=11,
                          fontweight='bold')
        ax_bar.set_xlim(0, max(vis) * 1.15 if vis else 0.1)
        ax_bar.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for i, (vi, pid) in enumerate(zip(vis, pids)):
            ax_bar.text(vi + 0.002, i, f'{vi:.4f}', va='center', fontsize=7)

        # Legend for bar chart
        bar_legend = [
            mpatches.Patch(color=TEAM_COLORS[0], label=TEAM_NAMES[0]),
            mpatches.Patch(color=TEAM_COLORS[1], label=TEAM_NAMES[1]),
        ]
        ax_bar.legend(handles=bar_legend, loc='lower right', fontsize=8)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {output_path}')
        plt.close(fig)
    else:
        plt.show()


def visualize_all_frames(
    csv_path: str,
    output_dir: str,
    step: int = 5,
    show_bar_chart: bool = True
):
    """
    Generate visualizations for every Nth frame.

    Args:
        csv_path: Path to CSV
        output_dir: Directory to save images
        step: Generate every Nth frame
        show_bar_chart: Include bar chart panel
    """
    # Get all frame numbers
    frames_set = set()
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames_set.add(int(row['帧号']))

    all_frames = sorted(frames_set)
    selected_frames = all_frames[::step]

    print(f'Generating {len(selected_frames)} visualizations '
          f'(every {step}th frame, {all_frames[0]}-{all_frames[-1]})...')

    os.makedirs(output_dir, exist_ok=True)

    for fn in selected_frames:
        out_path = os.path.join(output_dir, f'voronoi_frame_{fn:04d}.png')
        visualize_frame(csv_path, fn, out_path, show_bar_chart)

    print(f'Done! Images saved to: {output_dir}')


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(base_dir),
                            'player_final_real_coords_segment_0001.csv')
    output_dir = os.path.join(base_dir, 'voronoi_frames')

    # Visualize all frames (every 10th for speed)
    visualize_all_frames(csv_path, output_dir, step=10, show_bar_chart=True)

    # Also visualize a specific frame for quick check
    print('\n--- Single frame example ---')
    single_out = os.path.join(output_dir, 'example_frame_004.png')
    visualize_frame(csv_path, 4, single_out, show_bar_chart=True)
