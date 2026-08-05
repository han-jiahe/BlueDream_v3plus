#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将球员追踪 CSV 转换为 samples.json 格式（按帧分组，嵌套 players 数组），并添加两个球门节点。"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np


def _open_text_with_fallback(path: str) -> Tuple[TextIO, str]:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            f = open(path, "r", encoding=enc, newline="")
            f.read(2048)
            f.seek(0)
            return f, enc
        except UnicodeDecodeError as e:
            last_error = e
    raise UnicodeDecodeError(
        "unknown", b"", 0, 1,
        f"无法解码文件：{path}，已尝试编码：{', '.join(encodings)}。最后错误：{last_error}",
    )


def build_samples(
    rows: List[Dict[str, str]],
    fps: float,
    field_corners: List[List[float]],
) -> List[Dict[str, Any]]:
    frames: Dict[int, List[Dict[str, Any]]] = {}
    frame_order: List[int] = []

    # 第一步：按帧收集球员（不含球门）
    for row in rows:
        frame_num = int(row["帧号"])
        team_value = int(row["队伍"])

        if team_value not in (0, 1):
            continue  # 跳过裁判等其他目标

        team = "home" if team_value == 0 else "away"

        player = {
            "pixel": {
                "x": float(row["画面X"]),
                "y": float(row["画面Y"]),
            },
            "bbox": "NULL",
            "track_id": int(row["球员ID"]),
            "team": team,
            "field": {
                "x": float(row["真实X(米)"]),
                # Flip Y: CSV uses top-left origin (image convention)
                # → JSON uses bottom-left origin (standard math coords)
                "y": 68.0 - float(row["真实Y(米)"]),
            },
        }
        if frame_num not in frames:
            frames[frame_num] = []
            frame_order.append(frame_num)
        frames[frame_num].append(player)

    # 定义两个球门的像素坐标（基于提供的球场四角），注意这里只是为补全数据结构，不可使用！！！
    # 四角: 左上(192,108), 右上(1728,108), 右下(1728,972), 左下(192,972)
    left_goal_pixel = (192, (108 + 972) // 2)      # (192, 540)
    right_goal_pixel = (1728, (108 + 972) // 2)    # (1728, 540)
    left_goal_field = (0.0, 34.0)                  # 左球门真实坐标（米）
    right_goal_field = (105.0, 34.0)               # 右球门真实坐标（米）

    # 第二步：为每一帧添加两个球门节点
    for frame_num, players in frames.items():
        # 收集 home 和 away 球员的 field 坐标（忽略 referee）
        home_positions = []
        away_positions = []
        for p in players:
            if p["team"] == "home":
                home_positions.append((p["field"]["x"], p["field"]["y"]))
            elif p["team"] == "away":
                away_positions.append((p["field"]["x"], p["field"]["y"]))

        home_centroid = np.mean(home_positions, axis=0) if home_positions else None
        away_centroid = np.mean(away_positions, axis=0) if away_positions else None

        def assign_goal(goal_field_pos):
            # 计算到两队质心的距离，距离近者得
            dist_home = (
                np.linalg.norm(np.array(goal_field_pos) - home_centroid)
                if home_centroid is not None
                else float("inf")
            )
            dist_away = (
                np.linalg.norm(np.array(goal_field_pos) - away_centroid)
                if away_centroid is not None
                else float("inf")
            )
            # 如果两队质心都不存在（无任何 home/away 球员），默认左门归 home，右门归 away
            if home_centroid is None and away_centroid is None:
                return "home" if goal_field_pos == left_goal_field else "away"
            return "home" if dist_home <= dist_away else "away"

        left_team = assign_goal(left_goal_field)
        right_team = assign_goal(right_goal_field)

        left_goal = {
            "pixel": {"x": left_goal_pixel[0], "y": left_goal_pixel[1]},
            "bbox": "NULL",
            "track_id": -1,
            "team": left_team,
            "field": {"x": left_goal_field[0], "y": left_goal_field[1]},
        }
        right_goal = {
            "pixel": {"x": right_goal_pixel[0], "y": right_goal_pixel[1]},
            "bbox": "NULL",
            "track_id": -2,
            "team": right_team,
            "field": {"x": right_goal_field[0], "y": right_goal_field[1]},
        }

        players.append(left_goal)
        players.append(right_goal)

    # 第三步：构建 samples 列表
    samples: List[Dict[str, Any]] = []
    for idx, frame_num in enumerate(frame_order):
        timestamp = idx / fps
        players = frames[frame_num]
        samples.append({
            "timestamp": timestamp,
            "frame_number": frame_num,
            "sample_number": idx,
            "frame_file": f"football_analysis_output\\frames\\frame_{idx:06d}.jpg",
            "players": players,
            "num_players": len(players),          # 包括两个球门
            "field_corners": field_corners,
        })
    return samples


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="将球员追踪 CSV 转换为 samples.json 格式（按帧分组），并添加两个球门节点。"
    )
    p.add_argument(
        "-i", "--input",
        default="player_final_real_coords.csv",
        help="输入 CSV 路径（默认：player_final_real_coords.csv）",
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        help="输出 JSON 路径（默认：与输入同名 .json）",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="帧率，用于计算 timestamp（默认：15）",
    )
    p.add_argument(
        "--field-corners",
        type=str,
        default="192,108;1728,108;1728,972;192,972",
        help="场角坐标（像素），格式: x1,y1;x2,y2;x3,y3;x4,y4",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="美化输出（缩进 + 换行）",
    )
    return p


def parse_corners(raw: str) -> List[List[float]]:
    corners: List[List[float]] = []
    for part in raw.split(";"):
        x, y = part.strip().split(",")
        corners.append([float(x.strip()), float(y.strip())])
    if len(corners) != 4:
        raise ValueError("field_corners 必须包含 4 个点，用分号分隔")
    return corners


def main() -> int:
    args = build_arg_parser().parse_args()

    input_path = args.input
    if args.output is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".json"
    else:
        output_path = args.output

    field_corners = parse_corners(args.field_corners)

    in_file, enc = _open_text_with_fallback(input_path)
    try:
        reader = csv.DictReader(in_file)
        rows: List[Dict[str, str]] = list(reader)
    finally:
        in_file.close()

    samples = build_samples(rows, args.fps, field_corners)

    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(
            samples, out_file,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )

    print(f"OK: {input_path} ({enc}) -> {output_path}")
    print(f"  共 {len(samples)} 帧, 第一个帧号={samples[0]['frame_number'] if samples else 'N/A'}, 最后一个帧号={samples[-1]['frame_number'] if samples else 'N/A'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())