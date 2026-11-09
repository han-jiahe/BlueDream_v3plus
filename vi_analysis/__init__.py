# BlueDream v3plus — VI Analysis Module
# =========================================================================
# 论文口径（arXiv:2102.11182）：聚类 Variation of Information（单位 bit）
#   - clustering.py      ：聚类 + 公式(2) 总 VI（vi）+ 公式(3) 球员/节点 VI（vi_per_node）
#   - clustering_vi.py   ：逐时刻 VI 率（滚动窗口；总 / 球员 / 球队）
#   - vi_analysis.py     ：compute_vi_distribution（滚动窗口 VI 率，读写 JSON）
#
# 附加项（非论文 VI，仅空间可视化/探索，平时研究不用）：
#   - voronoi_vi.py      ：Voronoi 影响力（单元格面积占比）
# =========================================================================

from .clustering import (
    hierarchical_clustering_by_nearest_neighbor,
    vi,
    vi_per_node,
    visualize_clusters,
)
from .clustering_vi import compute_vi_rates, compute_vi_rates_to_json
from .vi_analysis import compute_vi_distribution
from .voronoi_vi import compute_voronoi_vi_from_csv, compute_frame_vi

__all__ = [
    # 论文口径（聚类 VI）
    "hierarchical_clustering_by_nearest_neighbor",
    "vi",
    "vi_per_node",
    "visualize_clusters",
    "compute_vi_rates",
    "compute_vi_rates_to_json",
    "compute_vi_distribution",
    # 附加项（Voronoi 影响力）
    "compute_voronoi_vi_from_csv",
    "compute_frame_vi",
]
