import math
import numpy as np
import cv2
import random

def euclidean_distance(p1, p2):
    """计算两点之间的欧氏距离（基于球场坐标）"""
    point1 = p1["field"]
    point2 = p2["field"]
    return math.sqrt((point1["x"] - point2["x"])**2 + (point1["y"] - point2["y"])**2)

def find_nearest_neighbor(p, dataset):
    """找到点p的最近邻（除自身外）"""
    min_dist = float('inf')
    nearest_point = p
    for q in dataset:
        if p == q:
            continue
        dist = euclidean_distance(p, q)
        if dist < min_dist:
            min_dist = dist
            nearest_point = q
    return min_dist, nearest_point

def hierarchical_clustering_by_nearest_neighbor(sample):
    """
    基于最近邻的层次聚类算法
    输入：单个采样点字典（包含players列表）
    输出：簇字典，键为簇ID，值为该簇内的球员对象列表
    """
    players = sample["players"]
    clusters = {}          # 球员ID -> 簇ID
    cluster_id_counter = 0

    for p in players:
        player_id = p["track_id"]  # 获取球员ID
        _, nearest_p = find_nearest_neighbor(p, players)
        nearest_id = nearest_p["track_id"]  # 获取最近邻球员ID
        
        p_cluster = clusters.get(player_id)
        nearest_cluster = clusters.get(nearest_id)

        if p_cluster is None and nearest_cluster is None:
            # 两者都未分配，新建簇
            new_id = cluster_id_counter
            clusters[player_id] = new_id
            clusters[nearest_id] = new_id
            cluster_id_counter += 1
        elif nearest_cluster is None:
            # nearest_p未分配，加入p的簇
            clusters[nearest_id] = p_cluster
        elif p_cluster is None:
            # p未分配，加入nearest_p的簇
            clusters[player_id] = nearest_cluster
        elif p_cluster != nearest_cluster:
            # 两者在不同簇，合并两个簇
            for pid, cid in list(clusters.items()):
                if cid == nearest_cluster:
                    clusters[pid] = p_cluster

    # 整理成最终格式：簇ID -> 球员对象列表
    final_clusters = {}
    for p in players:
        cid = clusters.get(p["track_id"])
        if cid is not None:
            final_clusters.setdefault(cid, []).append(p)
    
    return final_clusters


def print_clusters_info(clusters):
    """打印聚类信息"""
    print(f"总共形成了 {len(clusters)} 个簇:")
    for cid, points in clusters.items():
        print(f"簇 {cid}: 包含 {len(points)} 个球员(门)")
        ids = [p["track_id"] for p in points]
        print(f"  IDs: {ids}")

def vi(cluster_A, cluster_B, n_nodes):
    """
    计算两个聚类结果之间的 Variation of Information (VI)
    cluster_A, cluster_B: 簇字典，格式同 hierarchical_clustering_by_nearest_neighbor 的输出
    n_nodes: 总节点数（即该帧的球员+球门总数）
    """
    k = len(cluster_A)
    l = len(cluster_B)
    rij = np.zeros((k, l))

    # 将簇列表转换为列表形式方便索引
    A_clusters = list(cluster_A.values())
    B_clusters = list(cluster_B.values())

    for i in range(k):
        for j in range(l):
            # 计算交集大小
            intersect = 0
            for pa in A_clusters[i]:
                for pb in B_clusters[j]:
                    if pa["track_id"] == pb["track_id"]:
                        intersect += 1
                        break  # 每个track_id只匹配一次
            rij[i, j] = intersect / n_nodes

    pi = np.sum(rij, axis=1)  # 行和
    qj = np.sum(rij, axis=0)  # 列和

    vi_val = 0.0
    for i in range(k):
        for j in range(l):
            if rij[i, j] > 0:
                term1 = rij[i, j] * np.log2(rij[i, j] / pi[i]) if pi[i] > 0 else 0
                term2 = rij[i, j] * np.log2(rij[i, j] / qj[j]) if qj[j] > 0 else 0
                vi_val -= (term1 + term2)
    return vi_val

def visualize_clusters(frame, clusters, output_path=None):
    """
    在图像上可视化聚类结果，每个簇用不同颜色标记
    frame: 图像数组 (BGR)
    clusters: 簇字典
    output_path: 若指定，保存图像
    """
    # 为每个簇生成随机颜色
    colors = {}
    for cid in clusters:
        colors[cid] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    for cid, players in clusters.items():
        color = colors[cid]
        for player in players:
            if 'pixel' in player:
                cx, cy = int(player['pixel']['x']), int(player['pixel']['y'])
                cv2.circle(frame, (cx, cy), 8, color, -1)
                cv2.putText(frame, str(cid), (cx-10, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    if output_path:
        cv2.imwrite(output_path, frame)
    return frame
