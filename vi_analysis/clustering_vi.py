# -*- coding: utf-8 -*-
"""
聚类 Variation of Information（论文 arXiv:2102.11182）时间序列计算。

实现论文定义的“信息变差率”（VI-dot）：
  - 总 VI       ：两个聚类之间的 Variation of Information（bit，论文公式(2)/label eq:3）
  - 球员/节点 VI：把簇对贡献均分给交集中的节点（bit，论文公式(3)/label eq:4）
  - 球队 VI     ：该队 11 名场上球员 + 该队球门节点 的 VI 贡献之和（bit）
  - VI 率       ：VI / 时间间隔（bit/s）

滑动窗口语义（论文口径，逐时刻、滚动、重叠）：
  每个时刻 t 的 VI 率 = VI(C_{t-lag}, C_t) / (time_t - time_{t-lag})，
  即“该时刻往前 lag 个采样点”的两个端点聚类之间的 VI 除以时间。
  每个时刻都有一个值；不是把时间轴切成不相交窗口再取平均。
  论文原始设定 lag=1（相邻采样点；10Hz 下 dt=0.1s）。
"""

import json
import os
from collections import defaultdict

import numpy as np

from .clustering import (
    hierarchical_clustering_by_nearest_neighbor,
    vi,
    vi_per_node,
)


def _cluster_sample(sample):
    return hierarchical_clustering_by_nearest_neighbor(sample)


def _team_map_of(sample):
    """返回 {track_id: team}，以当前采样点的归属为准（含球门节点）。"""
    team_of = {}
    for p in sample["players"]:
        team_of[p["track_id"]] = p.get("team", "unknown")
    return team_of


def compute_vi_rates(samples, lag=1):
    """
    计算逐时刻（滚动窗口）的 VI 率序列。

    Args:
        samples: samples.json 读入的 list；每个元素含 timestamp / players / ...
        lag: 窗口长度（采样点数）。lag=1 即论文原始“相邻聚类”口径；
             例如 25fps 下 lag=100 表示 4 秒窗口。

    Returns:
        list[dict]，每个元素对应一个时刻 t（窗口末端），键：
          - timestamp : 该时刻（秒）
          - frame_index: 采样点下标 t
          - lag       : 窗口长度（采样点数）
          - dt        : 时间间隔（秒）
          - total_vi  : 总 VI（bit，未除以时间）
          - vi_rate   : 总 VI 率（bit/s）
          - player_vi : {track_id: VI 率}（每节点，含球门）
          - team_vi   : {team: VI 率}（每队，含球门节点）

    节点数 n 取“该窗口两个端点帧的节点并集”（完整 11v11 数据下恒为 24 =
    22 球员 + 2 球门）；单侧缺失的节点按单元素簇补入，故换人/漏检不丢概率质量。

    自检恒等式（每时刻成立）：
          sum(player_vi) ≈ vi_rate
          sum(team_vi.values()) ≈ vi_rate
          （home + away = 总 VI 率；不再是“恒为 1.000”）
    """
    if lag < 1:
        raise ValueError("lag 必须 >= 1")
    if len(samples) < 2:
        return []

    clusters = [_cluster_sample(s) for s in samples]

    results = []
    for t in range(lag, len(samples)):
        c_before = clusters[t - lag]
        c_after = clusters[t]
        dt = samples[t]["timestamp"] - samples[t - lag]["timestamp"]
        if dt <= 0:
            # 时间戳未递增时的退化保护（正常数据不会发生）
            dt = 1.0

        total_vi = vi(c_before, c_after)
        per_node = vi_per_node(c_before, c_after)

        team_of = _team_map_of(samples[t])

        player_vi_rate = {}
        team_vi = defaultdict(float)
        for tid, contrib in per_node.items():
            rate = float(contrib) / dt
            player_vi_rate[tid] = rate
            team_vi[team_of.get(tid, "unknown")] += rate

        results.append({
            "timestamp": samples[t]["timestamp"],
            "frame_index": t,
            "lag": lag,
            "dt": dt,
            "total_vi": total_vi,
            "vi_rate": total_vi / dt,
            "player_vi": player_vi_rate,
            "team_vi": dict(team_vi),
        })

    return results


def compute_vi_rates_to_json(samples, output_json_path, lag=1):
    """计算 VI 率序列并保存为 JSON（附带自检）。"""
    results = compute_vi_rates(samples, lag=lag)
    for r in results:
        s_player = sum(r["player_vi"].values())
        s_team = sum(r["team_vi"].values())
        assert abs(s_player - r["vi_rate"]) < 1e-6, (
            f"自检失败：sum(player_vi)={s_player} != vi_rate={r['vi_rate']}")
        assert abs(s_team - r["vi_rate"]) < 1e-6, (
            f"自检失败：sum(team_vi)={s_team} != vi_rate={r['vi_rate']}")
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"VI 率序列已保存至 {output_json_path}（{len(results)} 个时刻）")
    return results


# ============================================================
# 自检（验证公式(2)/(3) 与滚动窗口语义）
# ============================================================

def _mk_node(tid, team, x, y):
    return {"track_id": tid, "team": team, "field": {"x": x, "y": y}}


def _selfcheck():
    import math

    # ---- 1) 相同聚类 → VI = 0，每个节点贡献 0 ----
    A = {0: [_mk_node(1, "home", 0, 0), _mk_node(2, "away", 10, 0)]}
    B = {0: [_mk_node(1, "home", 0, 0), _mk_node(2, "away", 10, 0)]}
    assert abs(vi(A, B)) < 1e-12, "相同聚类应 VI=0"
    assert all(abs(v) < 1e-12 for v in vi_per_node(A, B).values())

    # ---- 2) 一个 4 节点簇裂成两个 2 节点簇 → VI = 1 bit，每节点 0.25 bit ----
    A = {0: [_mk_node(i, "home" if i % 2 else "away", i, 0) for i in range(1, 5)]}
    B = {0: [_mk_node(1, "home", 1, 0), _mk_node(2, "away", 2, 0)],
         1: [_mk_node(3, "home", 3, 0), _mk_node(4, "away", 4, 0)]}
    total = vi(A, B)
    assert abs(total - 1.0) < 1e-9, f"4 裂成 2+2 应 VI=1，实际 {total}"
    contrib = vi_per_node(A, B)
    assert all(abs(v - 0.25) < 1e-9 for v in contrib.values()), contrib

    # ---- 2b) 节点消失（换人/漏检）→ 单元素簇补入，VI > 0，不丢概率质量 ----
    A = {0: [_mk_node(1, "home", 1, 0), _mk_node(2, "home", 2, 0),
             _mk_node(3, "home", 3, 0)]}
    B = {0: [_mk_node(1, "home", 1, 0), _mk_node(2, "home", 2, 0)]}
    total = vi(A, B)
    expected = -((2.0 / 3.0) * math.log2(2.0 / 3.0)
                 + (1.0 / 3.0) * math.log2(1.0 / 3.0))
    assert abs(total - expected) < 1e-9, f"节点消失应 VI={expected}，实际 {total}"
    per = vi_per_node(A, B)
    assert set(per.keys()) == {1, 2, 3}, "消失节点也应有贡献"
    assert abs(sum(per.values()) - total) < 1e-9
    assert all(v >= 0 for v in per.values()), "节点贡献应非负"

    # ---- 3) 每节点贡献之和 == 总 VI（随机打乱也不破坏恒等式） ----
    for _ in range(50):
        ids = list(range(1, 7))
        rng = np.random.default_rng(12345 + _)
        a_assign = {tid: int(rng.integers(0, 3)) for tid in ids}
        b_assign = {tid: int(rng.integers(0, 3)) for tid in ids}
        A = {}
        B = {}
        for tid in ids:
            A.setdefault(a_assign[tid], []).append(_mk_node(tid, "home", tid, 0))
            B.setdefault(b_assign[tid], []).append(_mk_node(tid, "home", tid, 0))
        tot = vi(A, B)
        per = vi_per_node(A, B)
        assert abs(sum(per.values()) - tot) < 1e-9, (tot, per)

    # ---- 4) 滚动窗口语义：输出长度 = N - lag，dt 由 timestamp 差给出 ----
    samples = []
    for t in range(6):
        samples.append({
            "timestamp": t * 0.1,
            "players": [
                _mk_node(1, "home", 0.0, 0.0),
                _mk_node(2, "away", 10.0 + t, 0.0),
                _mk_node(-1, "home", 0.0, 34.0),   # 左球门
                _mk_node(-2, "away", 105.0, 34.0),  # 右球门
            ],
        })
    rates = compute_vi_rates(samples, lag=2)
    assert len(rates) == 6 - 2, f"应输出 {6-2} 个时刻，实际 {len(rates)}"
    assert abs(rates[0]["dt"] - 0.2) < 1e-12
    for r in rates:
        assert abs(sum(r["player_vi"].values()) - r["vi_rate"]) < 1e-9
        assert abs(sum(r["team_vi"].values()) - r["vi_rate"]) < 1e-9
        assert abs(sum(r["team_vi"].values()) * r["dt"] - r["total_vi"]) < 1e-9

    print("clustering_vi self-check passed (all assertions OK)")


if __name__ == "__main__":
    # 本模块使用包内相对导入，请以 `python -m vi_analysis.clustering_vi` 运行自检
    _selfcheck()
