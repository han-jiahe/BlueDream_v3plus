# -*- coding: utf-8 -*-
"""
聚类 Variation of Information 的逐时刻（滚动窗口）VI 率计算。

论文口径（arXiv:2102.11182）：
  每个时刻 t 的 VI 率 = VI(C_{t-lag}, C_t) / (time_t - time_{t-lag})，
  逐时刻滑动、每时刻都有一个值；不是把时间轴切成不相交窗口再取窗口内平均。
"""

import json
import os

from .clustering_vi import compute_vi_rates


def compute_vi_distribution(samples_json_path, output_json_path,
                            window_size=100, save_cluster_images=False):
    """
    计算逐时刻（滚动窗口）的 VI 率序列，保存到 JSON。

    Args:
        samples_json_path: samples.json 路径（video_processor / csv2json 产出）
        output_json_path: 输出 JSON 路径
        window_size: 窗口长度（采样点数）。25fps 下 100 帧 = 4 秒；
                     论文原始口径为 lag=1（相邻聚类之间）。
        save_cluster_images: 是否保存每帧聚类可视化（可选，需 cv2）

    Returns:
        list[dict]，每个元素对应一个时刻 t（窗口末端），键：
          timestamp / frame_index / lag / dt / total_vi / vi_rate /
          player_vi / team_vi
    """
    with open(samples_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if save_cluster_images:
        _save_cluster_images(data, samples_json_path, output_json_path)

    results = compute_vi_rates(data, lag=window_size)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"VI 率序列已保存至 {output_json_path}（{len(results)} 个时刻）")
    return results


def _save_cluster_images(data, samples_json_path, output_json_path):
    """（可选）保存每帧聚类可视化。仅在 save_cluster_images=True 时调用。"""
    import cv2
    from .clustering import hierarchical_clustering_by_nearest_neighbor, visualize_clusters

    cluster_img_dir = os.path.join(os.path.dirname(output_json_path), 'cluster_frames')
    os.makedirs(cluster_img_dir, exist_ok=True)
    base_dir = os.path.dirname(samples_json_path)

    for i, sample in enumerate(data):
        clusters_t = hierarchical_clustering_by_nearest_neighbor(sample)
        frame_path = sample.get("frame_file", "")
        if not os.path.isabs(frame_path):
            frame_path = os.path.join(base_dir, frame_path)
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            if frame is not None:
                out_path = os.path.join(cluster_img_dir, f'cluster_{i:06d}.jpg')
                visualize_clusters(frame, clusters_t, out_path)
