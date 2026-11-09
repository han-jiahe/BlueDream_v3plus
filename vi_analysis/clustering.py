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

def _node_to_cluster_index(clusters):
    """返回 {track_id: 簇ID} 映射。"""
    node_to_cid = {}
    for cid, members in clusters.items():
        for m in members:
            node_to_cid[m["track_id"]] = cid
    return node_to_cid


def _contingency(cluster_A, cluster_B):
    """
    构建两个聚类之间的列联表（论文公式(2) 的 r_ij、p_i、q_j）。

    两个聚类被视为对"两帧出现过节点并集"的划分：若某节点只出现在其中一个
    聚类里（换人 / 漏检 / 该帧缺检测），则在另一个聚类里作为单元素簇补入，
    使"出现/消失"的节点也正确计入信息变差，而不是静默丢失概率质量。

    返回 (rij, pi, qj, node_pair, cnt)：
      - cnt[i, j]  ：|x_i ∩ y_j|（交集节点数，精确整数）
      - rij[i, j]  ：cnt[i, j] / n，n = 两聚类节点并集大小（论文 r_ij；
                     完整 11v11 数据下并集恒为 24 = 22 球员 + 2 球门）
      - pi, qj     ：rij 的行和、列和（论文 p_i、q_j，均按并集归一，和为 1）
      - node_pair  ：{track_id: (行索引 i, 列索引 j)}，覆盖并集中的每个节点
    """
    node_to_A = _node_to_cluster_index(cluster_A)
    node_to_B = _node_to_cluster_index(cluster_B)
    union = sorted(set(node_to_A) | set(node_to_B))
    n = len(union)

    def _index_map(node_to_cid, cids):
        cid_to_idx = {cid: i for i, cid in enumerate(sorted(cids))}
        m = {}
        next_i = len(cid_to_idx)
        for tid in union:
            if tid in node_to_cid:
                m[tid] = cid_to_idx[node_to_cid[tid]]
            else:
                m[tid] = next_i  # 缺失节点补为单元素簇
                next_i += 1
        return m, next_i

    A_map, k = _index_map(node_to_A, cluster_A.keys())
    B_map, l = _index_map(node_to_B, cluster_B.keys())

    cnt = np.zeros((k, l))
    node_pair = {}
    for tid in union:
        i = A_map[tid]
        j = B_map[tid]
        cnt[i, j] += 1
        node_pair[tid] = (i, j)

    rij = cnt / n if n > 0 else np.zeros_like(cnt)
    pi = rij.sum(axis=1)
    qj = rij.sum(axis=0)
    return rij, pi, qj, node_pair, cnt


def vi(cluster_A, cluster_B):
    """
    计算两个聚类结果之间的 Variation of Information (VI)（论文公式(2)，label eq:3）

        VI(X;Y) = -Σ_ij r_ij [log2(r_ij/p_i) + log2(r_ij/q_j)]   （bit，恒 ≥ 0）

    cluster_A, cluster_B: 簇字典，格式同 hierarchical_clustering_by_nearest_neighbor 的输出。
    节点数 n 取两个聚类的节点并集（完整 11v11 下 = 24 = 22 球员 + 2 球门），
    单侧缺失的节点按单元素簇补入，因此换人/漏检也会被正确计入。
    """
    rij, pi, qj, _, _ = _contingency(cluster_A, cluster_B)
    vi_val = 0.0
    k, l = rij.shape
    for i in range(k):
        for j in range(l):
            r = rij[i, j]
            if r > 0:
                # r > 0 时必有 pi[i] >= r > 0 且 qj[j] >= r > 0，不会出现 log(0)
                vi_val -= r * (np.log2(r / pi[i]) + np.log2(r / qj[j]))
    return float(vi_val)


def vi_per_node(cluster_A, cluster_B):
    """
    计算每个节点（球员/球门）在两聚类之间的 VI 贡献（论文公式(3)，label eq:4）：

        VI^{p_ij} = -r_ij * [log2(r_ij/p_i) + log2(r_ij/q_j)] / |x_i ∩ y_j|

    即把簇对 (x_i, y_j) 对总 VI 的贡献均分给交集中的每个节点。贡献恒 ≥ 0：
    因为 r_ij ≤ p_i 且 r_ij ≤ q_j，两个 log 项 ≤ 0，取负后非负（无论簇合并还是
    分裂都为正信息变差）。单侧缺失的节点按单元素簇补入（|x_i ∩ y_j| = 1），
    所以换人/漏检节点也会得到其应得的贡献。
    所有节点贡献之和 == vi()（总 VI）。
    """
    rij, pi, qj, node_pair, cnt = _contingency(cluster_A, cluster_B)
    contrib = {}
    for tid, (i, j) in node_pair.items():
        intersect_size = int(cnt[i, j])  # = |x_i ∩ y_j|（精确整数，且 ≥ 1）
        if intersect_size > 0:
            r = rij[i, j]
            term = r * (np.log2(r / pi[i]) + np.log2(r / qj[j]))
            contrib[tid] = float(-term / intersect_size) + 0.0  # +0.0 归一化 -0.0
        else:
            contrib[tid] = 0.0
    return contrib

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
