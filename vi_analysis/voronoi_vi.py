# -*- coding: utf-8 -*-
"""
Spatial Voronoi VI (Voronoi Influence) computation for soccer analysis.
Pitch dimensions: 105m × 68m (FIFA standard)

Computes for each frame:
  - Individual VI: player's Voronoi cell area / total pitch area (7140 m^2)
  - Team VI: sum of all team members' individual VIs

Method:
  1. For each frame, collect player coordinates in pitch space (meters)
  2. Add boundary mirror points to bound cells within [0,105] x [0,68]
  3. Compute Voronoi tessellation via scipy.spatial.Voronoi
  4. Clip each cell polygon to pitch rectangle using shapely
  5. VI = clipped_area / 7140.0
"""

import csv
import json
import os
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from collections import defaultdict

# ============================================================
# Pitch constants (FIFA standard)
# ============================================================
PITCH_LENGTH = 105.0   # meters (touchline)
PITCH_WIDTH  = 68.0    # meters (goal line)
PITCH_AREA   = PITCH_LENGTH * PITCH_WIDTH  # 7140.0 m^2


# ============================================================
# Voronoi VI computation
# ============================================================

def add_boundary_mirrors(points: np.ndarray) -> np.ndarray:
    """
    Add mirror points across all four pitch boundaries to ensure
    Voronoi cells of interior points are bounded within the pitch.

    For each original point (x, y) we create 8 mirror points:
      - Left:   (-x, y)
      - Right:  (2*L - x, y)
      - Bottom: (x, -y)
      - Top:    (x, 2*W - y)
      - Four corner mirrors

    Args:
        points: (N, 2) array of player coordinates in meters

    Returns:
        (M, 2) array with original + mirror points
    """
    mirrors = []
    L = PITCH_LENGTH
    W = PITCH_WIDTH

    for x, y in points:
        mirrors.append([-x, y])              # left
        mirrors.append([2*L - x, y])         # right
        mirrors.append([x, -y])              # bottom
        mirrors.append([x, 2*W - y])         # top
        mirrors.append([-x, -y])             # bottom-left
        mirrors.append([2*L - x, -y])        # bottom-right
        mirrors.append([-x, 2*W - y])        # top-left
        mirrors.append([2*L - x, 2*W - y])   # top-right

    all_pts = np.vstack([points, np.array(mirrors)])
    return all_pts


def compute_frame_vi(players: list) -> dict:
    """
    Compute Voronoi VI for a single frame.

    Args:
        players: list of dicts with keys 'player_id', 'x', 'y', 'team'

    Returns:
        dict with:
          - 'player_vi': {player_id: vi_value}
          - 'team_vi': {team_code: total_vi}
          - 'cell_polygons': {player_id: Polygon} (for visualization)
    """
    n = len(players)
    if n == 0:
        return {'player_vi': {}, 'team_vi': {}, 'cell_polygons': {}}

    # Original points
    orig_points = np.array([[p['x'], p['y']] for p in players])

    # Add boundary mirrors
    all_points = add_boundary_mirrors(orig_points)

    # Compute Voronoi
    vor = Voronoi(all_points)

    pitch_box = box(0, 0, PITCH_LENGTH, PITCH_WIDTH)

    player_vi = {}
    team_vi = defaultdict(float)
    cell_polygons = {}

    for i, player in enumerate(players):
        region_idx = vor.point_region[i]
        region_vertices_idx = vor.regions[region_idx]

        if not region_vertices_idx or -1 in region_vertices_idx:
            # Unbounded region (should not happen with mirror points, but be safe)
            player_vi[player['player_id']] = 0.0
            cell_polygons[player['player_id']] = None
            continue

        # Build polygon from Voronoi vertices
        polygon_vertices = vor.vertices[region_vertices_idx]
        cell_poly = Polygon(polygon_vertices)

        # Clip to pitch boundaries
        try:
            clipped = cell_poly.intersection(pitch_box)
            if clipped.is_empty:
                area = 0.0
                cell_polygons[player['player_id']] = None
            else:
                area = clipped.area
                cell_polygons[player['player_id']] = clipped
        except Exception:
            area = 0.0
            cell_polygons[player['player_id']] = None

        vi = area / PITCH_AREA
        player_vi[player['player_id']] = vi
        team_vi[player['team']] += vi

    return {
        'player_vi': player_vi,
        'team_vi': dict(team_vi),
        'cell_polygons': cell_polygons
    }


def compute_voronoi_vi_from_csv(
    csv_path: str,
    output_json_path: str = None,
    include_referee: bool = False
) -> dict:
    """
    Main entry point: read CSV player coordinates and compute Voronoi VI
    for every frame.

    CSV expected columns (UTF-8 with BOM handled):
        帧号, 球员ID, 画面X, 画面Y, 雷达X, 雷达Y, 真实X(米), 真实Y(米), 队伍

    Args:
        csv_path: Path to the CSV file
        output_json_path: Optional path to save VI results (JSON)
        include_referee: If False, exclude team 3 (referee)

    Returns:
        dict: all results keyed by frame number
    """
    # Read CSV
    frames = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = int(row['队伍'])
            if not include_referee and team == 3:
                continue
            frames[int(row['帧号'])].append({
                'player_id': int(row['球员ID']),
                'x': float(row['真实X(米)']),
                # Flip Y: CSV uses top-left origin → VI uses bottom-left origin
                'y': 68.0 - float(row['真实Y(米)']),
                'team': team
            })

    results = {}
    for frame_num in sorted(frames.keys()):
        players = frames[frame_num]
        frame_data = compute_frame_vi(players)

        # Keep only serializable fields for JSON output
        results[frame_num] = {
            'player_vi': frame_data['player_vi'],
            'team_vi': frame_data['team_vi'],
            'num_players': len(players),
            'cell_polygons': frame_data['cell_polygons']  # kept in-memory for viz
        }

    # Save JSON summary (without polygons — they are not JSON-serializable)
    if output_json_path:
        summary = {
            'pitch': {'length_m': PITCH_LENGTH, 'width_m': PITCH_WIDTH,
                       'area_m2': PITCH_AREA},
            'total_frames': len(results),
            'frames': {}
        }
        for fn, data in results.items():
            summary['frames'][str(fn)] = {
                'player_vi': data['player_vi'],
                'team_vi': data['team_vi'],
                'num_players': data['num_players']
            }
        os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f'VI results saved to: {output_json_path}')

    return results


# ============================================================
# Convenience: generate summary stats
# ============================================================

def print_summary(results: dict) -> None:
    """Print a summary table of VI results across all frames."""
    if not results:
        print('No results.')
        return

    frames = sorted(results.keys())
    teams = set()
    for fn in frames:
        teams.update(results[fn]['team_vi'].keys())

    print(f'\n{"="*60}')
    print(f'Voronoi VI Summary  |  Pitch: {PITCH_LENGTH}m x {PITCH_WIDTH}m'
          f'  |  Area: {PITCH_AREA} m^2')
    print(f'{"="*60}')
    print(f'{"Frame":>6s}  {"Players":>8s}', end='')
    for t in sorted(teams):
        print(f'  {"Team " + str(t) + " VI":>12s}', end='')
    print(f'  {"Sum VI":>10s}')
    print(f'{"-"*60}')

    total_team_vi = defaultdict(float)
    for fn in frames:
        n = results[fn]['num_players']
        tvi = results[fn]['team_vi']
        print(f'{fn:6d}  {n:8d}', end='')
        for t in sorted(teams):
            val = tvi.get(t, 0.0)
            print(f'  {val:12.6f}', end='')
            total_team_vi[t] += val
        print(f'  {sum(tvi.values()):10.6f}')

    print(f'{"-"*60}')
    print(f'{"AVG":>6s}  {"":>8s}', end='')
    for t in sorted(teams):
        print(f'  {total_team_vi[t]/len(frames):12.6f}', end='')
    print()
    print(f'{"="*60}')

    # Per-player average VI
    print(f'\nAverage individual VI (top 10):')
    player_vi_sum = defaultdict(float)
    player_vi_count = defaultdict(int)
    for fn in frames:
        for pid, vi in results[fn]['player_vi'].items():
            player_vi_sum[pid] += vi
            player_vi_count[pid] += 1

    player_avg = [(pid, player_vi_sum[pid]/player_vi_count[pid])
                  for pid in player_vi_sum]
    player_avg.sort(key=lambda x: -x[1])
    for pid, avg_vi in player_avg[:10]:
        print(f'  Player {pid:3d}: {avg_vi:.6f}')


# ============================================================
# Run standalone
# ============================================================
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(base_dir),
                            'player_final_real_coords_segment_0001.csv')
    output_json = os.path.join(base_dir, 'voronoi_vi_results.json')

    results = compute_voronoi_vi_from_csv(csv_path, output_json)
    print_summary(results)
