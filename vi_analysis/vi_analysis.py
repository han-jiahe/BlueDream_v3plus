import json
import os
from matplotlib.pylab import sample
from tqdm import tqdm
from clustering import hierarchical_clustering_by_nearest_neighbor, vi, visualize_clusters
import cv2

def compute_vi_distribution(samples_json_path, output_json_path, window_size=100, save_cluster_images=False):
    """
    计算每个时间窗口的平均VI，并保存到JSON文件
    Args:
        samples_json_path: samples.json 的路径
        output_json_path: 输出VI分布JSON的路径
        window_size: 窗口大小（帧数），默认100帧（4秒@25fps）
        save_cluster_images: 是否保存每帧的聚类可视化图像
    """
    # 读取采样数据
    with open(samples_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sample_number = len(data)
    print(f"总采样点数: {sample_number}")
    print(f"视频时长: {data[-1]['timestamp']:.1f}秒")

    # 计算窗口数量
    total_windows = (sample_number - window_size) // window_size + 1
    print(f"窗口大小: {window_size}帧, 总窗口数: {total_windows}")

    # 存储所有帧的聚类结果，用于窗口内计算
    all_clusters = []  # 每个元素为字典：{'clusters':..., 'timestamp':..., 'num_players':..., 'cluster_file':...}

    # 创建输出目录（如果保存聚类图像）
    if save_cluster_images:
        cluster_img_dir = os.path.join(os.path.dirname(output_json_path), 'cluster_frames')
        os.makedirs(cluster_img_dir, exist_ok=True)

    # 第一步：对每一帧进行聚类，并可选保存可视化图像
    print("正在对每一帧进行聚类...")
    for i in tqdm(range(sample_number), desc="聚类帧"):
        sample = data[i]
        clusters_t = hierarchical_clustering_by_nearest_neighbor(sample)
        timepoint = sample["timestamp"]
        player_count = sample["num_players"]

        cluster_file = None
        if save_cluster_images:
            # 读取原始帧图像
            base_dir = os.path.dirname(samples_json_path)
            frame_path = os.path.join(base_dir, sample["frame_file"])   # 已调整
            #   frame_path = sample["frame_file"]   注意：samples.json中保存的是相对路径，可能需要调整
            if not os.path.isabs(frame_path):
                # 假设frame_file相对于samples.json所在目录
                base_dir = os.path.dirname(samples_json_path)
                frame_path = os.path.join(base_dir, frame_path)
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    # 可视化聚类
                    out_path = os.path.join(cluster_img_dir, f'cluster_{i:06d}.jpg')
                    visualize_clusters(frame, clusters_t, out_path)
                    cluster_file = out_path
                else:
                    print(f"警告：无法读取图像 {frame_path}")
            else:
                print(f"警告：图像文件不存在 {frame_path}")

        all_clusters.append({
            "clusters": clusters_t,
            "timestamp": timepoint,
            "num_players": player_count,
            "cluster_file": cluster_file
        })

    # 第二步：滑动窗口计算平均VI
    vi_results = []
    print("正在计算窗口平均VI...")
    for start in tqdm(range(0, sample_number - window_size + 1, window_size), total=total_windows, desc="处理时间窗口"):
        end = start + window_size
        window_vi_sum = 0.0
        pair_count = 0
        # 窗口中间时间（用于标记）
        mid_idx = start + window_size // 2
        window_time = all_clusters[mid_idx]["timestamp"]

        # 计算窗口内所有帧对之间的VI（归一化时间差）
        for i in range(start, end):
            for j in range(i+1, end):
                time_diff = abs(all_clusters[i]["timestamp"] - all_clusters[j]["timestamp"])
                if time_diff == 0:
                    continue  # 理论上不会发生
                vi_val = vi(all_clusters[i]["clusters"], all_clusters[j]["clusters"], all_clusters[i]["num_players"])
                window_vi_sum += vi_val / time_diff
                pair_count += 1

        avg_vi = window_vi_sum / pair_count if pair_count > 0 else 0.0
        vi_results.append({
            "window_time": window_time,
            "average_vi": avg_vi
        })

    # 保存VI分布到JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(vi_results, f, indent=2, ensure_ascii=False)
    print(f"VI分布已保存至 {output_json_path}")
    return vi_results