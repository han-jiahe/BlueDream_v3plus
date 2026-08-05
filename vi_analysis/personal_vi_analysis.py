#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Personal VI Analysis: compute average Voronoi VI per player across all frames.
Generates ranking, distribution chart, and detailed report.
"""

import json
import os
import csv
from collections import defaultdict
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


def load_voronoi_vi_results(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_player_team_mapping(csv_path: str) -> dict:
    player_team = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row['球员ID'])
            team = int(row['队伍'])
            if pid not in player_team:
                player_team[pid] = team
    return player_team


def compute_personal_vi(results: dict, player_team: dict) -> dict:
    player_data = defaultdict(list)
    frames = results.get('frames', {})
    for frame_num, frame_data in frames.items():
        for pid_str, vi_val in frame_data.get('player_vi', {}).items():
            player_data[int(pid_str)].append(vi_val)

    personal_vi = {}
    for pid, vi_list in player_data.items():
        vi_arr = np.array(vi_list)
        personal_vi[pid] = {
            'player_id': pid,
            'team': player_team.get(pid, -1),
            'mean_vi': float(np.mean(vi_arr)),
            'std_vi': float(np.std(vi_arr)),
            'max_vi': float(np.max(vi_arr)),
            'min_vi': float(np.min(vi_arr)),
            'frame_count': len(vi_list),
        }
    return personal_vi


def compute_team_avg_vi(personal_vi: dict) -> dict:
    team_data = defaultdict(list)
    for pid, info in personal_vi.items():
        team_data[info['team']].append(info['mean_vi'])
    result = {}
    for team, vi_list in team_data.items():
        vi_arr = np.array(vi_list)
        result[team] = {
            'avg_vi': float(np.mean(vi_arr)),
            'std_vi': float(np.std(vi_arr)),
            'players': len(vi_list),
        }
    return result


def print_personal_vi_report(personal_vi: dict, team_avg: dict) -> None:
    sorted_players = sorted(personal_vi.values(), key=lambda x: -x['mean_vi'])
    sep = "=" * 80
    print(sep)
    print(" Personal Voronoi VI Report")
    print(" VI = player Voronoi cell area / total pitch area (7140 m^2)")
    print(sep)

    print(f"\n{'Team':>6s}  {'Players':>8s}  {'Avg VI':>10s}  {'VI Std':>10s}")
    print("-" * 40)
    for team in sorted(team_avg.keys()):
        t = team_avg[team]
        print(f"{'Team '+str(team):>6s}  {t['players']:8d}  {t['avg_vi']:10.6f}  {t['std_vi']:10.6f}")

    print(f"\n{'Rank':>4s}  {'Player':>6s}  {'Team':>4s}  {'Mean VI':>10s}  {'Std':>10s}  {'Max':>10s}  {'Min':>10s}  {'Frames':>8s}")
    print("-" * 80)
    for rank, p in enumerate(sorted_players, 1):
        print(f"{rank:4d}  {p['player_id']:6d}  {p['team']:4d}  "
              f"{p['mean_vi']:10.6f}  {p['std_vi']:10.6f}  "
              f"{p['max_vi']:10.6f}  {p['min_vi']:10.6f}  "
              f"{p['frame_count']:8d}")

    all_vi = [p['mean_vi'] for p in sorted_players]
    print(f"\n--- Stats ---")
    print(f"  Total players: {len(sorted_players)}")
    print(f"  Top VI: Player {sorted_players[0]['player_id']} = {sorted_players[0]['mean_vi']:.6f}")
    print(f"  Bottom VI: Player {sorted_players[-1]['player_id']} = {sorted_players[-1]['mean_vi']:.6f}")
    print(f"  Overall mean VI: {np.mean(all_vi):.6f}")
    print(sep)


def plot_personal_vi(personal_vi: dict, output_path: str) -> None:
    if not MATPLOTLIB_OK:
        print("  [SKIP] matplotlib not available")
        return

    sorted_players = sorted(personal_vi.values(), key=lambda x: -x['mean_vi'])
    player_ids = [str(p['player_id']) for p in sorted_players]
    mean_vis = [p['mean_vi'] for p in sorted_players]
    std_vis = [p['std_vi'] for p in sorted_players]
    teams = [p['team'] for p in sorted_players]

    colors = ['#2196F3' if t == 0 else '#F44336' if t == 1 else '#9E9E9E' for t in teams]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    ax1.bar(range(len(player_ids)), mean_vis, yerr=std_vis,
            color=colors, edgecolor='white', linewidth=0.5,
            capsize=3, error_kw={'linewidth': 0.8})
    ax1.set_xticks(range(len(player_ids)))
    ax1.set_xticklabels(player_ids, fontsize=8)
    ax1.set_ylabel('Mean Voronoi VI', fontsize=12)
    ax1.set_title('Personal Voronoi VI Ranking (Mean +/- Std)', fontsize=14, fontweight='bold')
    ax1.axhline(y=np.mean(mean_vis), color='gray', linestyle='--', linewidth=1,
                label=f'Overall Avg: {np.mean(mean_vis):.4f}')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='Team 0'),
        Patch(facecolor='#F44336', label='Team 1'),
        Patch(facecolor='#9E9E9E', label='Referee'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax2 = axes[1]
    team_groups = defaultdict(list)
    for p in sorted_players:
        team_groups[p['team']].append(p['mean_vi'])

    team_order = sorted(team_groups.keys())
    box_data = [team_groups[t] for t in team_order]
    box_labels = [f'Team {t}\n(n={len(team_groups[t])})' for t in team_order]

    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='black', markersize=6))

    box_colors = ['#2196F3', '#F44336', '#9E9E9E']
    for patch, color in zip(bp['boxes'], box_colors[:len(team_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel('Mean Voronoi VI', fontsize=12)
    ax2.set_title('Team-level VI Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout(pad=2)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {output_path}")


def save_personal_vi_json(personal_vi: dict, team_avg: dict, output_path: str) -> None:
    sorted_players = sorted(personal_vi.values(), key=lambda x: -x['mean_vi'])
    output = {
        'pitch': {'length_m': 105.0, 'width_m': 68.0, 'area_m2': 7140.0},
        'description': 'Personal Voronoi VI: average Voronoi cell area / total pitch area per player',
        'team_summary': {str(k): v for k, v in team_avg.items()},
        'players_ranked': sorted_players
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Personal VI JSON saved: {output_path}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vi_json_path = os.path.join(base_dir, 'voronoi_vi_results.json')
    csv_path = os.path.join(os.path.dirname(base_dir),
                            'player_final_real_coords_segment_0001.csv')

    print("[1/4] Loading Voronoi VI data...")
    vi_results = load_voronoi_vi_results(vi_json_path)
    print(f"  Total frames: {vi_results['total_frames']}")

    print("[2/4] Loading player-team mapping...")
    player_team = load_player_team_mapping(csv_path)
    print(f"  Players found: {len(player_team)}")

    print("[3/4] Computing personal VI stats...")
    personal_vi = compute_personal_vi(vi_results, player_team)
    team_avg = compute_team_avg_vi(personal_vi)
    print_personal_vi_report(personal_vi, team_avg)

    print("[4/4] Saving results...")
    output_dir = os.path.join(base_dir, 'football_analysis_output', 'data')
    personal_vi_json = os.path.join(output_dir, 'personal_vi.json')
    save_personal_vi_json(personal_vi, team_avg, personal_vi_json)

    personal_vi_png = os.path.join(output_dir, 'personal_vi_chart.png')
    plot_personal_vi(personal_vi, personal_vi_png)

    print("\nPersonal VI analysis complete!")
