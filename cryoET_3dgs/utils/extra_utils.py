import torch
import numpy as np
import open3d as o3d



def o3d_knn(pts, num_knn):
    '''
    使用open3d库来计算给定点集的k近邻(KNN)
    输入参数: 
    pts: 一个N*3的数组 表示N个3D点
    num_knn: k值
    '''
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd) # KD树，用于加速最近邻查询
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1) # 查找k近邻。返回每个点的k个邻居的索引和距离
        indices.append(i[1:])
        sq_dists.append(d[1:]) # 平方距离
    return np.array(sq_dists), np.array(indices)


def weighted_l2_loss_v2(x, y, w):
    '''
    计算加权的L2损失。
    '''
    # return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()

    return (((x - y) ** 2).sum(-1) * w + 1e-10).mean()


def count_zero_voxels(fields: torch.Tensor):
    """
    统计体素化结果中值为0的体素数量及占比。
    
    参数:
        fields (torch.Tensor): 体素化结果，通常是一个 3D 或 4D 张量。

    返回:
        zero_count (int): 体素值为 0 的体素数量
        total_count (int): 总体素数量
        zero_ratio (float): 体素值为 0 的比例
    """

    # 去掉 batch/channel 维度（如有）
    if fields.dim() == 5:
        # e.g., [1, 1, D, H, W]
        fields = fields.squeeze()

    # 计算统计
    zero_count = torch.sum(fields == 0).item()
    total_count = fields.numel()
    zero_ratio = zero_count / total_count

    print(f"\n******* Total voxels: {total_count}")
    print(f"        Zero-value voxels: {zero_count}")
    print(f"        Zero voxel ratio: {zero_ratio:.4f}")
    return zero_count, total_count, zero_ratio



import torch
import matplotlib.pyplot as plt

def voxel_density_statistics(fields: torch.Tensor, plot_hist: bool = True, bins: int = 50):
    """
    统计体素密度值的基本统计信息，并可选绘制直方图。

    参数:
        fields (torch.Tensor): 体素化结果张量
        plot_hist (bool): 是否绘制密度直方图
        bins (int): 直方图分桶数

    返回:
        stats (dict): 包含非零体素的统计量
    """
    if fields.dim() == 5:
        fields = fields.squeeze()

    # 获取非零体素值
    nonzero_fields = fields[fields > 0]

    if nonzero_fields.numel() == 0:
        print("No nonzero voxels found.")
        return {}

    stats = {
        "count": nonzero_fields.numel(),
        "min": nonzero_fields.min().item(),
        "max": nonzero_fields.max().item(),
        "mean": nonzero_fields.mean().item(),
        "std": nonzero_fields.std().item(),
        "median": nonzero_fields.median().item(),
        "quantile_25": nonzero_fields.quantile(0.25).item(),
        "quantile_75": nonzero_fields.quantile(0.75).item(),
        "quantile_95": nonzero_fields.quantile(0.95).item()
    }

    print("Voxel Density Statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:.6f}")

    # 可视化直方图
    if plot_hist:
        nonzero_np = nonzero_fields.cpu().numpy()
        plt.figure(figsize=(6, 4))
        plt.hist(nonzero_np, bins=bins, color='skyblue', edgecolor='black')
        plt.title("Voxel Density Histogram")
        plt.xlabel("Density Value")
        plt.ylabel("Voxel Count")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    return stats


def get_cuda_memory_usage(device=None):
    device = device or torch.cuda.current_device()
    memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 单位 MB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return memory_allocated, memory_reserved
