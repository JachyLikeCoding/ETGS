import torch
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import seaborn as sns
import os, sys
from mayavi import mlab
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from cryoET_3dgs.utils.system_utils import mkdir_p
from cryoET_3dgs.utils.graphics_utils import BasicPointCloud
from cryoET_3dgs.utils.general_utils import inverse_sigmoid, inverse_softplus, build_rotation, build_scaling_rotation, strip_symmetric, get_expon_lr_func
from simple_knn._C import distCUDA2
from plyfile import PlyData, PlyElement
from cryoET_3dgs.scene.deformation_et import deform_network


def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def inverse_log_transform(y):
    return torch.sign(y) * (torch.expm1(torch.abs(y)))


class GaussianModel_cryoET:
    def setup_functions(self):
        # 定义构建3D高斯协方差矩阵的函数
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2) # 计算实际的协方差矩阵
            symm = strip_symmetric(actual_covariance)   # 提取对称部分
            return symm #最终返回对称的协方差矩阵

        if self.disable_xyz_log_activation:
            self.xyz_activation = lambda x: x
            self.inverse_xyz_activation = lambda x: x
        else:
            self.xyz_activation = inverse_log_transform
            self.inverse_xyz_activation = log_transform

        # 初始化一些激活函数
        # if self.scale_bound is not None:
        #     scale_min_bound, scale_max_bound = self.scale_bound
        #     print('\n[debug] scale min bound:', scale_min_bound, ', max bound:', scale_max_bound)
        #     assert(
        #         scale_min_bound < scale_max_bound
        #     ), "Error: scale min must < scale max."
        #     self.scaling_activation = (lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound) + scale_min_bound)
        #     self.scaling_inverse_activation = lambda x: inverse_sigmoid(torch.relu((x - scale_min_bound)/(scale_max_bound - scale_min_bound)))

        # scale_min = 0.2
        # scale_max = 3.0

        # self.scaling_activation = lambda x: torch.sigmoid(x) * (scale_max - scale_min) + scale_min
        # self.scaling_inverse_activation = lambda x: torch.log(
        #     torch.clamp((x - scale_min) / (scale_max - scale_min), 1e-6, 1 - 1e-6) /
        #     (1 - torch.clamp((x - scale_min) / (scale_max - scale_min), 1e-6, 1 - 1e-6))
        # )

        # self.scaling_activation = torch.exp # 用指数函数将尺度限制为非负数
        # self.scaling_inverse_activation = torch.log
        self.scaling_activation = torch.nn.functional.softplus  # 使用softplus函数限制尺度为正数，但增长更平缓
        self.scaling_inverse_activation = lambda x: torch.log(torch.exp(x) - 1)  # 对应softplus的逆函数
        
        self.covariance_activation = build_covariance_from_scaling_rotation # 协方差矩阵的激活函数
        
        self.intensity_activation = torch.sigmoid # 将强度限制在0-1范围
        self.intensity_inverse_activation = inverse_sigmoid # 强度的逆激活函数

        self.rotation_activation = torch.nn.functional.normalize #用于归一化旋转矩阵的函数
        
        self.deformation_activation = torch.nn.Softplus() 
        self.deformation_inverse_activation = inverse_softplus

    
    def __init__(self, disable_xyz_log_activation, args=None, use_deformation=False, scale_bound=None):
        self.use_deform = use_deformation
        self.disable_xyz_log_activation = disable_xyz_log_activation
        
        if self.use_deform:
            self._deformation = deform_network(W=args.net_width, D=args.deform_depth, 
                                                min_embeddings=args.min_embeddings, 
                                                max_embeddings=args.max_embeddings,
                                                num_tilts=args.total_num_tilts,
                                                args=args)
            self._embedding = torch.empty(0)
        else:
            self._deformation = None
            self._embedding = None

        self._xyz = torch.empty(0) # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0) # 旋转参数，用四元数表示
        self._intensity = torch.empty(0) # 密度   
        self.max_radii2D = torch.empty(0)   # 投影到2D时，每个2D gaussian最大的半径
        self.xyz_gradient_accum = torch.empty(0)  # 3D gaussian中心位置的累计梯度
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 1.0
        self.scale_bound = scale_bound
        self.split_num = 0
        self.clone_num = 0
        
        self.setup_functions()


    def capture(self):
        if self.use_deform:
            return (
                self._xyz,
                self._deformation.state_dict(),
                self._intensity,
                self._scaling,
                self._rotation,
                self._embedding,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self.scale_bound,
            )
        return (
                self._xyz,
                self._intensity,
                self._scaling,
                self._rotation,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self.scale_bound,
            )


    def restore(self, model_args, training_args):
        if self.use_deform:
            (self._xyz,
            self._deformation,
            self._intensity,
            self._scaling,
            self._rotation,
            self.max_radii2D,
            self._embedding,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.scale_bound) = model_args
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
        else:
            (self._xyz,
            self._intensity,
            self._scaling,
            self._rotation,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.scale_bound) = model_args
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
            


    @property
    def get_scaling(self):
        # return self._scaling
        # print(f"Max scaling before activation: {self._scaling.max()}, Min scaling before activation: {self._scaling.min()}")
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        # print(f"Max rotation before normalization: {self._rotation.max()}, Min rotation before normalization: {self._rotation.min()}")
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_intensity(self):
        # return self._intensity
        return self.intensity_activation(self._intensity)
    
    @property
    def get_embedding(self):
        return self._embedding
    
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    
    def reset_intensity(self, reset_intensity=0.01):
        intensities_new = self.intensity_inverse_activation(torch.min(self.get_intensity, torch.ones_like(self.get_intensity) * reset_intensity))
        optimizable_tensors = self.replace_tensor_to_optimizer(intensities_new, "intensity")
        self._intensity = optimizable_tensors["intensity"]
        print('\n[debug] reset intensity: ', reset_intensity)


    def create_random_gaussians(self, intensity_threshold:float, spatial_lr_scale:float, image_width:int, image_height:int, 
                                grid_size:int, volume_x:int, volume_y:int, volume_z:int, factor:float, scale_init:float):
        self.spatial_lr_scale = spatial_lr_scale
        x_coords = torch.linspace(0, volume_x, steps=volume_x//grid_size, device="cuda")
        y_coords = torch.linspace(0, volume_y, steps=volume_y//grid_size, device="cuda")
        z_coords = torch.linspace(0, volume_z, steps=volume_z//grid_size, device="cuda")
        # 生成均匀分布的3D网格点
        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        sampled_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
        print(f"\nGrid_size = {grid_size}")
        print(f"Number of grid points: {sampled_points.shape[0]}")

        # Initialize intensities
        scale_x, scale_y = image_width/volume_x, image_height/volume_y
        scale_z = scale_y

        center = torch.tensor([volume_x * 0.5, volume_y * 0.5, volume_z * 0.5], device="cuda")
        sampled_points -= center
        sampled_points[:, 0] *= scale_x
        sampled_points[:, 1] *= scale_y
        sampled_points[:, 2] *= scale_z

        sampled_intensities = torch.full((sampled_points.shape[0],), intensity_threshold, device="cuda")
        scales = torch.full((sampled_points.shape[0],3), scale_init, device="cuda") # (P, 3)
        scales *= factor

        # 初始化每个点的旋转参数为单元四元数（无旋转）
        rots = torch.zeros((sampled_points.shape[0], 4), device="cuda") # (P, 4)
        rots[:, 0] = 1 # 四元数的实部为1，表示无旋转
 
        self._xyz = nn.Parameter(sampled_points.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._intensity = nn.Parameter(sampled_intensities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 存储2D投影的最大半径，初始化为0
        print(f"Initialized {sampled_points.shape[0]} Gaussian points with fixed intensity {intensity_threshold} "
          f"and fixed scale value {scale_init}.")
        

    # # 获得初始点
    # def create_from_wbp(self, wbp: BasicPointCloud, intensity_threshold:float, init_size:int, spatial_lr_scale:float, image_width:int, image_height:int, grid_size:int,
    #                     volume_x:int, volume_y:int, volume_z:int, factor:float):
    #     """
    #     从粗重建数据初始化模型参数
    #     :param wbp: wBP粗重建数据, 包含点的位置和密度值
    #     :param spatial_lr_scale: 空间学习率缩放因子，影响位置参数的学习率
    #     """
    #     self.spatial_lr_scale = spatial_lr_scale
    #     points = torch.tensor(np.asarray(wbp.points)).float().cuda() # (P, 3)

    #     intensities = torch.tensor(np.asarray(wbp.intensities)).float().cuda()
    #     intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
    #     # intensities = self.intensity_activation(intensities)
    #     # intensities = inverse_sigmoid(intensities) # (P, 1)
    #     intensities = torch.clamp(intensities, 1e-4, 1-1e-4)
    #     intensities = torch.log(intensities / (1 - intensities))  # inverse sigmoid

    #     print('\n[debug]: intensities.shape:', intensities.shape)
    #     print("\nNumber of points at initialisation : ", points.shape[0])
    #     print('intensities max:', max(intensities), '| min:', min(intensities))

    #     # Filter points based on intensity threshold
    #     tmp = intensities
    #     high_intensity_mask = tmp.squeeze() >  intensity_threshold
    #     high_intensity_points = points[high_intensity_mask]
    #     high_intensity_values = intensities[high_intensity_mask]
    #     print("Number of points after filter: ", high_intensity_points.shape[0])
    #     num_points = high_intensity_points.shape[0]

    #     # Randomly sample points from high intensity points
    #     if init_size > num_points:
    #         init_size = num_points

    #     # 划分网格进行采样
    #     min_coords = high_intensity_points.min(dim=0)[0]
    #     max_coords = high_intensity_points.max(dim=0)[0]
    #     print('\nmax coords:', max_coords, '\nmin coords:', min_coords)
    #     grid_coords = ((high_intensity_points - min_coords) / grid_size).floor().long()
    #     unique_grid_coords, inverse_indices = torch.unique(grid_coords, return_inverse=True, dim=0)
    #     selected_indices = []

    #     # 在每个网格单元中选择intensity最大的点
    #     for i in range(unique_grid_coords.shape[0]):
    #         cell_indices = (inverse_indices == i).nonzero(as_tuple=True)[0]
    #         cell_intensities = high_intensity_values[cell_indices]
    #         max_intensity_idx = torch.argmax(cell_intensities)
    #         selected_indices.append(cell_indices[max_intensity_idx])


    #     selected_indices = torch.tensor(selected_indices, dtype=torch.long).cuda()
    #     sampled_points = high_intensity_points[selected_indices]
    #     sampled_intensities = high_intensity_values[selected_indices]
    #     print("Number of points after grid select: ", sampled_points.shape[0])

    #     scale_x, scale_y = image_width / volume_x, image_height / volume_y
    #     scale_z = scale_y

    #     center = torch.tensor([volume_x * 0.5, volume_y * 0.5, volume_z * 0.5], device="cuda")
    #     sampled_points -= center
    #     sampled_points[:, 0] *= scale_x
    #     sampled_points[:, 1] *= scale_y
    #     sampled_points[:, 2] *= -scale_z

    #     target_volume_z = 512
    #     z_factor = target_volume_z / volume_z

    #     # 计算每个点到其最近的K个点的平均距离的平方，用于确定高斯的尺度参数
    #     dist = torch.sqrt(torch.clamp_min(distCUDA2(sampled_points), 1e-6)) # (P, )

    #     # from torch_cluster import knn_graph

    #     # edge_index = knn_graph(sampled_points, k=8, loop=False)
    #     # distances = torch.norm(sampled_points[edge_index[0]] - sampled_points[edge_index[1]], dim=1)
    #     # avg_dist = torch.zeros_like(sampled_points[:, 0])
    #     # avg_dist.index_add_(0, edge_index[0], distances)
    #     # dist = avg_dist / 8

    #     if self.scale_bound is not None:
    #         dist = torch.clamp(
    #             dist, self.scale_bound[0] + 1e-6, self.scale_bound[1] - 1e-6
    #         )

    #     scales = self.scaling_inverse_activation(dist)[...,None].repeat(1, 3) # (P, 3)
    #     scales *= factor

    #     # 初始化每个点的旋转参数为单元四元数（无旋转）
    #     rots = torch.zeros((sampled_points.shape[0], 4), device="cuda") # (P, 4)
    #     rots[:, 0] = 1 # 四元数的实部为1，表示无旋转
    #     print('\n[debug]: sampled_intensities.shape:', sampled_intensities.shape)

    #     # zero initialization
    #     if self.use_deform:
    #         embedding = torch.zeros((sampled_points.shape[0], self._deformation.gaussian_embedding_dim)).float().cuda()
    #         self._embedding = nn.Parameter(embedding.requires_grad_(True))
    #         self._deformation = self._deformation.to("cuda")


    #     # 将以上计算的参数设置为模型的可训练参数
    #     self._xyz = nn.Parameter(sampled_points.requires_grad_(True)) #[p,3]
    #     # self._deformation = self._deformation.to("cuda")
    #     # self._embedding = nn.Parameter(embedding.requires_grad_(True))
    #     self._scaling = nn.Parameter(scales.requires_grad_(True)) # [p,4]
    #     self._rotation = nn.Parameter(rots.requires_grad_(True)) # [p,4]
    #     self._intensity = nn.Parameter(sampled_intensities.requires_grad_(True)) #[p]
        
    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 存储2D投影的最大半径，初始化为0



    # 获得初始点
    def create_from_wbp(self, wbp: BasicPointCloud, intensity_threshold:float, init_size:int, spatial_lr_scale:float, image_width:int, image_height:int, grid_size:int,
                        volume_x:int, volume_y:int, volume_z:int, factor:float):
        """
        从粗重建数据初始化模型参数
        :param wbp: wBP粗重建数据, 包含点的位置和密度值
        :param spatial_lr_scale: 空间学习率缩放因子，影响位置参数的学习率
        """
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(np.asarray(wbp.points)).float().cuda() # (P, 3)

        intensities = torch.tensor(np.asarray(wbp.intensities)).float().cuda()
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())

        # intensities = inverse_sigmoid(intensities) # (P, 1)
        print('\n[debug]: intensities.shape:', intensities.shape)
        print("\nNumber of points at initialisation : ", points.shape[0])
        print('intensities max:', max(intensities), '| min:', min(intensities))

        # Filter points based on intensity threshold
        tmp = intensities
        high_intensity_mask = tmp.squeeze() < 1 - intensity_threshold
        high_intensity_points = points[high_intensity_mask]
        high_intensity_values = intensities[high_intensity_mask]
        print("Number of points after filter: ", high_intensity_points.shape[0])
        num_points = high_intensity_points.shape[0]

        # Randomly sample points from high intensity points
        if init_size > num_points:
            init_size = num_points

        # 划分网格进行采样
        min_coords = high_intensity_points.min(dim=0)[0]
        max_coords = high_intensity_points.max(dim=0)[0]
        print('\nmax coords:', max_coords, '\nmin coords:', min_coords)
        grid_coords = ((high_intensity_points - min_coords) / grid_size).floor().long()
        unique_grid_coords, inverse_indices = torch.unique(grid_coords, return_inverse=True, dim=0)
        selected_indices = []

        # 在每个网格单元中随机选择一个点
        for i in range(unique_grid_coords.shape[0]):
            cell_indices = (inverse_indices == i).nonzero(as_tuple=True)[0]
            random_index = cell_indices[torch.randint(len(cell_indices), (1,)).item()]
            selected_indices.append(random_index)

        selected_indices = torch.tensor(selected_indices, dtype=torch.long).cuda()
        # selected_indices = np.random.choice(num_points, size=init_size, replace=False)
        sampled_points = high_intensity_points[selected_indices]
        # sampled_intensities = high_intensity_values[selected_indices]
        # sampled_intensities = inverse_sigmoid(high_intensity_values[selected_indices])
        # sampled_intensities = self.intensity_inverse_activation(high_intensity_values[selected_indices])
        sampled_intensities = (high_intensity_values[selected_indices])
        
        print("Number of points after grid select: ", sampled_points.shape[0])

        scale_x, scale_y = image_width / volume_x, image_height / volume_y
        scale_z = scale_y

        center = torch.tensor([volume_x * 0.5, volume_y * 0.5, volume_z * 0.5], device="cuda")
        sampled_points -= center
        sampled_points[:, 0] *= scale_x
        sampled_points[:, 1] *= scale_y
        sampled_points[:, 2] *= -scale_z

        # 计算每个点到其最近的K个点的平均距离的平方，用于确定高斯的尺度参数
        dist = torch.sqrt(torch.clamp_min(distCUDA2(sampled_points), 1e-6)) # (P, )

        if self.scale_bound is not None:
            dist = torch.clamp(dist, self.scale_bound[0] + 1e-6, self.scale_bound[1] - 1e-6)

        scales = torch.log(dist)[...,None].repeat(1, 3) # (P, 3)
        scales *= factor
        # scales = self.scaling_inverse_activation(torch.full((sampled_points.shape[0], 3), 2.0, device="cuda"))  # Initialize scales with a constant size of 1.0

        # 初始化每个点的旋转参数为单元四元数（无旋转）
        rots = torch.zeros((sampled_points.shape[0], 4), device="cuda") # (P, 4)
        rots[:, 0] = 1
        print('\n[debug]: sampled_intensities.shape:', sampled_intensities.shape)

        if self.use_deform:
            embedding = torch.zeros((sampled_points.shape[0], self._deformation.gaussian_embedding_dim)).float().cuda()
            self._embedding = nn.Parameter(embedding.requires_grad_(True))
            self._deformation = self._deformation.to("cuda")

        # 将以上计算的参数设置为模型的可训练参数
        self._xyz = nn.Parameter(sampled_points.requires_grad_(True)) #[p,3]
        # self._deformation = self._deformation.to("cuda")
        # self._embedding = nn.Parameter(embedding.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True)) # [p,4]
        self._rotation = nn.Parameter(rots.requires_grad_(True)) # [p,4]
        self._intensity = nn.Parameter(sampled_intensities.requires_grad_(True)) #[p]
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 存储2D投影的最大半径，初始化为0


    def create_hierarchical_gaussian(self, wbp: BasicPointCloud, intensity_threshold:float, init_size:int, spatial_lr_scale:float, image_width:int, image_height:int, grid_size:int,
                        volume_x:int, volume_y:int, volume_z:int, factor:float, if_whitebg:bool=True):
        """
        从粗重建数据初始化模型参数，分层采样策略
        :param wbp: 粗重建数据, 包含点的位置和密度值
        :param spatial_lr_scale: 空间学习率缩放因子，影响位置参数的学习率
        """
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(np.asarray(wbp.points)).float().cuda() # (P, 3)
        
        intensities = torch.tensor(np.asarray(wbp.intensities)).float().cuda()

        # upper = torch.quantile(intensities.flatten(), 0.95)
        # print(f'intensities upper = {upper}')
        # intensities = intensities.clamp(min=0, max=upper)
        
        # intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
        # print('intensities max:', max(intensities), '| min:', min(intensities))

        # ==================== 高密度区域采样 ====================
        if if_whitebg: 
            high_intensity_mask = intensities.squeeze() < intensity_threshold
        else:
            high_intensity_mask = intensities.squeeze() > intensity_threshold
            
        high_intensity_points = points[high_intensity_mask]
        high_intensity_values = intensities[high_intensity_mask]
        print("High intensity points after filter: ", high_intensity_points.shape[0])

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(high_intensity_points.cpu().numpy())
        down_pcd = pcd.voxel_down_sample(voxel_size=grid_size)
        
        sampled_points = torch.tensor(np.asarray(down_pcd.points)).float().cuda()
        from scipy.spatial import cKDTree
        tree = cKDTree(high_intensity_points.cpu().numpy())
        _, idx = tree.query(np.asarray(down_pcd.points))
        # sampled_intensities = self.intensity_inverse_activation(high_intensity_values[idx])
        sampled_intensities = high_intensity_values[idx]
        print('sampled intensities max:', max(sampled_intensities), '| min:', min(sampled_intensities))

        # ==================== 背景区域均匀采样 ====================
        # 确定全体积边界
        volume_min = torch.tensor([0, 0, 0], device="cuda")
        volume_max = torch.tensor([volume_x, volume_y, volume_z], device="cuda")
        
        # 生成均匀背景网格（网格尺寸设为高密度区域的1.5倍）
        bg_grid_size = grid_size *  1.5
        x = torch.arange(volume_min[0]-15, volume_max[0]+15, bg_grid_size, device="cuda")
        y = torch.arange(volume_min[1]-volume_y//2, volume_max[1]+volume_y//2, bg_grid_size, device="cuda")
        z = torch.arange(volume_min[2]-15, volume_max[2]+15, bg_grid_size, device="cuda")
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        bg_grid_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
        # 过滤与高密度区域重叠的点（避免重复）
        kdtree = spatial.KDTree(sampled_points.detach().cpu().numpy())
        distances, _ = kdtree.query(bg_grid_points.detach().cpu().numpy(), k=1)
        bg_mask = distances > grid_size  # 保留与已有点距离超过grid_size的背景点
        bg_points = bg_grid_points[bg_mask]
        
        # 控制背景点数量（不超过高密度点的50%）
        # bg_num = min(len(bg_points), int(len(sampled_points)*0.5))
        # bg_points = bg_points[torch.randperm(len(bg_points))[:bg_num]]
        bg_num = len(bg_points)
        
        # 背景点强度初始化
        if if_whitebg:
            bg_val = torch.quantile(intensities, 0.95)  # 靠近白背景
            # bg_val = torch.tensor(1 - intensity_threshold)
        else:
            bg_val = torch.quantile(intensities, 0.05)  # 靠近黑背景
            # bg_val = torch.tensor(intensity_threshold)
        
        print(f'Background intensity value: {bg_val.item()}')
        bg_intensities = torch.full((bg_num,), bg_val.item(), device="cuda")

        # ==================== 合并两类点 ====================
        all_points = torch.cat([sampled_points, bg_points], dim=0)
        all_intensities = torch.cat([sampled_intensities, bg_intensities], dim=0)
        all_intensities = (all_intensities - all_intensities.min()) / (all_intensities.max() - all_intensities.min())

        # ==================== 参数初始化 ====================
        scale_x, scale_y = image_width / volume_x, image_height / volume_y
        scale_z = scale_y

        center = torch.tensor([volume_x * 0.5, volume_y * 0.5, volume_z * 0.5], device="cuda")
        all_points -= center
        all_points[:, 0] *= scale_x
        all_points[:, 1] *= scale_y
        all_points[:, 2] *= scale_z

        # 自适应尺度计算
        dist = torch.sqrt(torch.clamp_min(distCUDA2(all_points), 1e-6))
        if self.scale_bound is not None:
            dist = torch.clamp(dist, self.scale_bound[0]+1e-6, self.scale_bound[1]-1e-6)
        scales = torch.log(dist)[...,None].repeat(1, 3) * factor

        # 旋转初始化（背景点各向同性）
        rots = torch.zeros((len(all_points), 4), device="cuda")
        rots[:, 0] = 1
        # 背景点设置更大初始尺度
        # scales[len(sampled_points):] += torch.log(torch.tensor(3))  # 增大背景点尺寸

        if self.use_deform:
            embedding = torch.zeros((len(all_points), self._deformation.gaussian_embedding_dim)).float().cuda()
            self._embedding = nn.Parameter(embedding.requires_grad_(True))
            self._deformation = self._deformation.to("cuda")

        self._xyz = nn.Parameter(all_points.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._intensity = nn.Parameter(all_intensities.requires_grad_(True))
        self.max_radii2D = torch.zeros((len(all_points)), device="cuda")

        print(f"Total initialized Gaussians: {len(all_points)} (Signal {len(sampled_points)} + Background {bg_num})")
    
    
    
    
    def create_from_volume_gradient(self, volume: torch.Tensor, intensity_threshold: float, num_samples: int, spatial_lr_scale: float,
                                image_width: int, image_height: int, volume_x: int, volume_y: int, volume_z: int, factor: float,
                                start_fraction: float = 0.15, ini_sigma: float = 0.01, ini_intensity: float = 0.04):
        """
        从初始密度体图（如WBP重建）中提取梯度信息，初始化高斯参数。
        
        :param volume: (D, H, W) torch.Tensor, 初始密度体图
        """
        self.spatial_lr_scale = spatial_lr_scale
        W, H, D = volume.shape

        # 标准化并去除背景（air threshold）
        volume = volume.clone()
        volume[volume < intensity_threshold] = 0.0

        volume = volume.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

        # 计算体数据的梯度
        grad_x = torch.abs(volume[:, :, 1:-1, 1:-1, 1:-1] - volume[:, :, 1:-1, 1:-1, 2:])
        grad_y = torch.abs(volume[:, :, 1:-1, 1:-1, 1:-1] - volume[:, :, 1:-1, 2:, 1:-1])
        grad_z = torch.abs(volume[:, :, 1:-1, 1:-1, 1:-1] - volume[:, :, 2:, 1:-1, 1:-1])

        grad_norm = torch.sqrt(F.pad(grad_x, (1, 1, 1, 1, 1, 1)) ** 2 +
                            F.pad(grad_y, (1, 1, 1, 1, 1, 1)) ** 2 +
                            F.pad(grad_z, (1, 1, 1, 1, 1, 1)) ** 2)  # shape: [1,1,D,H,W]

        grad_norm = grad_norm.view(-1)
        start = int(start_fraction * D * H * W)
        _, indices = torch.topk(grad_norm, start + num_samples)
        indices = indices[start:]

        # 计算对应坐标
        coords = torch.stack(torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij'), dim=-1).reshape(-1, 3).cuda()
        sampled_coords = coords[indices]  # [N, 3]

        # 统计邻域数量以估算scale
        grid = torch.zeros((D, H, W), dtype=torch.int32, device="cuda")
        grid[sampled_coords[:, 0], sampled_coords[:, 1], sampled_coords[:, 2]] += 1

        kernel_size = 5
        padding = kernel_size // 2
        conv_kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device="cuda")
        neighbours_count = F.conv3d(grid.unsqueeze(0).unsqueeze(0).float(), conv_kernel, padding=padding).squeeze()
        num_neighbours = neighbours_count[sampled_coords[:, 0], sampled_coords[:, 1], sampled_coords[:, 2]]

        scaling = ini_sigma / num_neighbours.float()
        scaling = torch.log(scaling).unsqueeze(1).repeat(1, 3)

        # 获取采样点的强度并映射
        volume = volume.squeeze()
        intensities = ini_intensity * volume.reshape(-1)[indices] + 0.001
        intensities = self.intensity_inverse_activation(intensities).unsqueeze(1)

        # 归一化坐标 & 缩放
        sampled_coords = sampled_coords.float()
        sampled_coords -= torch.tensor([volume_x * 0.5, volume_y * 0.5, volume_z * 0.5], device="cuda")
        sampled_coords[:, 0] *= image_width / volume_x
        sampled_coords[:, 1] *= image_height / volume_y
        sampled_coords[:, 2] *= -image_height / volume_y  # cryoET中Z轴是反向的

        # 初始化旋转为单位四元数
        rotation = torch.zeros((num_samples, 4), device="cuda")
        rotation[:, 0] = 1

        # 设置为可学习参数
        self._xyz = nn.Parameter(sampled_coords.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._intensity = nn.Parameter(intensities.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))

        if self.use_deform:
            embedding = torch.zeros((num_samples, self._deformation.gaussian_embedding_dim), dtype=torch.float32, device="cuda")
            self._embedding = nn.Parameter(embedding.requires_grad_(True))
            self._deformation = self._deformation.to("cuda")

        self.max_radii2D = torch.zeros((num_samples,), device="cuda")

    
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'intensity']
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_deform:
            for i in range(self._embedding.shape[1]):
                l.append('embedding_{}'.format(i))
        return l
    


    def save_gaussianpoints(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        intensities = self._intensity.detach().cpu().numpy().reshape(-1, 1)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        if self.use_deform:
            embedding = np.round(self._embedding.detach().cpu().numpy(), 4)
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        print('\n[debug] xyz.shape', xyz.shape)
        print('[debug] intensities.shape', intensities.shape)
        print('[debug] scale.shape', scale.shape)
        print('[debug] rotation.shape', rotation.shape)
        
        if self.use_deform:
            attributes = np.concatenate((xyz, normals, intensities, scale, rotation, embedding), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, intensities, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def save_gaussianpoints_rgb(self, path):
        mkdir_p(os.path.dirname(path))

        # 将数据转为 numpy 数组并四舍五入保留 4 位小数
        xyz = self._xyz.detach().cpu().numpy().astype(np.float32)
        normals = np.zeros_like(xyz)  # 法线暂时不使用
        intensities = self.get_intensity.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)        
        scale = self.get_scaling.detach().cpu().numpy().astype(np.float32)
        print(f"\n ======save gaussians ====== Scale max: {scale.max():.4f}, Scale min: {scale.min():.4f}, Intensity max: {intensities.max():.4f}, Intensities min: {intensities.min():.4f}")

        rotation = self._rotation.detach().cpu().numpy().astype(np.float32)
        if self.use_deform:
            embedding = self._embedding.detach().cpu().numpy().astype(np.float32)

        # 将 intensity 归一化到 [0, 1] 范围，并映射为 RGB
        rgb_colors = self.intensity_to_rgb(intensities)
        # 计算不透明度（可以根据强度计算不透明度，或者根据缩放等）
        # opacity = self.intensity_to_opacity(intensities)
        opacity = intensities

        # 构建 PLY 文件的字段名
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), 
                    ('intensity', 'f4'),
                    ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), 
                    ('opacity', 'f4')]
        if self.use_deform:
            for i in range(embedding.shape[1]):
                dtype_full.append((f'embedding_{i}', 'f4'))

        # 添加 scale 和 rotation 信息
        for i in range(scale.shape[1]):  # 三维的 scale 属性
            dtype_full.append((f'scale_{i}', 'f4'))
        for i in range(rotation.shape[1]):  # 四元数的 rotation 属性
            dtype_full.append((f'rot_{i}', 'f4'))

        # 创建元素存储点的属性
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # 填充数据
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
        elements['intensity'] = intensities.flatten()  # 独立存储原始 intensity
        elements['red'] = rgb_colors[:, 0]
        elements['green'] = rgb_colors[:, 1]
        elements['blue'] = rgb_colors[:, 2]
        elements['f_dc_0'] = rgb_colors[:, 0]
        elements['f_dc_1'] = rgb_colors[:, 1]
        elements['f_dc_2'] = rgb_colors[:, 2]
        elements['opacity'] = opacity.flatten()

        if self.use_deform:
            for i in range(embedding.shape[1]):
                elements[f'embedding_{i}'] = embedding[:, i]
            
        # 填充缩放因子 (scale)，每个点有三个值
        for i in range(scale.shape[1]):
            elements[f'scale_{i}'] = scale[:, i]

        # 填充旋转矩阵 (rotation)，每个点有四个值
        for i in range(rotation.shape[1]):
            elements[f'rot_{i}'] = rotation[:, i]

        # 创建 PLY 元素
        el = PlyElement.describe(elements, 'vertex')

        # 保存为 PLY 文件
        PlyData([el]).write(path)



    def save_gaussianpoints_rgb2(self, path):
        mkdir_p(os.path.dirname(path))

        # 基本数据提取
        xyz = self._xyz.detach().cpu().numpy().astype(np.float32)
        normals = np.zeros_like(xyz)
        intensities = self.get_intensity.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
        scale = self.get_scaling.detach().cpu().numpy().astype(np.float32)
        rotation = self._rotation.detach().cpu().numpy().astype(np.float32)

        if self.use_deform:
            embedding = self._embedding.detach().cpu().numpy().astype(np.float32)

        # ===================== intensity 映射到 RGB & Alpha =====================
        # 将 intensity 归一化到 [0, 1]
        normalized_intensity = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)

        # intensity -> 灰度 RGB
        rgb_colors = (normalized_intensity * 255).astype(np.uint8).repeat(3, axis=1)
        # rgb_colors = self.intensity_to_colormap_rgb(intensities, cmap_name="jet")

        # intensity -> alpha 通道
        alpha_channel = (normalized_intensity * 255).astype(np.uint8)

        print(f"\n====== save gaussians ====== "
            f"Scale max: {scale.max():.4f}, Scale min: {scale.min():.4f}, "
            f"Intensity max: {intensities.max():.4f}, Intensity min: {intensities.min():.4f}")

        # ===================== 构建 ply 数据类型 =====================
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('intensity', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('alpha', 'u1')
        ]

        if self.use_deform:
            for i in range(embedding.shape[1]):
                dtype_full.append((f'embedding_{i}', 'f4'))

        for i in range(scale.shape[1]):
            dtype_full.append((f'scale_{i}', 'f4'))

        for i in range(rotation.shape[1]):
            dtype_full.append((f'rot_{i}', 'f4'))

        # ===================== 填充数据 =====================
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
        elements['intensity'] = intensities.flatten()
        elements['red'] = rgb_colors[:, 0]
        elements['green'] = rgb_colors[:, 1]
        elements['blue'] = rgb_colors[:, 2]
        elements['alpha'] = alpha_channel.flatten()

        if self.use_deform:
            for i in range(embedding.shape[1]):
                elements[f'embedding_{i}'] = embedding[:, i]

        for i in range(scale.shape[1]):
            elements[f'scale_{i}'] = scale[:, i]

        for i in range(rotation.shape[1]):
            elements[f'rot_{i}'] = rotation[:, i]

        # ===================== 写入 ply 文件 =====================
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)



    def intensity_to_colormap_rgb(self, intensity, cmap_name="viridis"):
        """
        将归一化后的 intensity 映射为 RGB 值，返回 uint8 类型数组。
        """
        norm_intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        colormap = cm.get_cmap(cmap_name)
        rgb_float = colormap(norm_intensity.flatten())[:, :3]  # Ignore alpha from colormap
        return (rgb_float * 255).astype(np.uint8)

    def intensity_to_rgb(self, intensity):
        """ 将 intensity 映射到热力图色阶 """
        # 归一化到 [0,1]
        intensity = intensity.squeeze()
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        # 使用 matplotlib 热力图色阶
        cmap = plt.get_cmap('hot')
        rgb = (cmap(intensity_norm)[:, :3] * 255).astype(np.uint8)
        return rgb

    def intensity_to_opacity(self, intensity):
        """ 根据 intensity 计算不透明度（可自定义曲线） """
        opacity = 1 / (1 + np.exp(-10 * (intensity - 0.5)))  # Sigmoid 函数增强对比度
        return opacity.astype(np.float32)


    def save_csv(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        intensities = self._intensity.detach().cpu().numpy().reshape(-1,1)
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        attributes = np.concatenate((xyz, normals, intensities, scale, rotation), axis=1)
        
        header = "x,y,z,nx,ny,nz,intensity,scale,rotation"
        np.savetxt(path, attributes, delimiter=',', header=header, comments='')


    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(), os.path.join(path, "cryoet_deformation.pth"))


    def load_ply(self, path):
        self.disable_xyz_log_activation = True
        self.xyz_activation = lambda x: x
        self.inverse_xyz_activation = lambda x: x
        path = path.replace('\\', "/")
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        intensities = np.asarray(plydata.elements[0]["intensity"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if self.use_deform:
            embedding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("embedding")]
            embedding_names = sorted(embedding_names, key = lambda x: int(x.split('_')[-1]))
            embeddings = np.zeros((xyz.shape[0], len(embedding_names)))
            for idx, attr_name in enumerate(embedding_names):
                embeddings[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self._embedding = nn.Parameter(torch.tensor(embeddings, dtype=torch.float, device="cuda").requires_grad_(True))


        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._intensity = nn.Parameter(torch.tensor(intensities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        print('self._intensity:', self._intensity.max(), self._intensity.min())
        print('gaussians number:', self._xyz.shape[0])
        print(f'********* [debug]: load ply {path} success...')


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def training_setup(self, training_args):
        """
        设置训练参数，包括初始化用于累计梯度的变量，配置优化器，以及创建学习率调度器
        """
        # 设置在训练过程中用于密集化处理的3D高斯点的比例
        self.percent_dense = training_args.percent_dense

        # 初始化用于累计3D高斯中心点位置梯度的张量，用于之后判断是否需要对3D高斯进行克隆或切分
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 配置各参数的优化器，包括指定参数、学习率和参数名称
        if self.use_deform:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name":"deformation"},
                {'params': [self._intensity], 'lr': training_args.intensity_lr_init * self.spatial_lr_scale, "name": "intensity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr_init * self.spatial_lr_scale, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr_init * self.spatial_lr_scale, "name": "rotation"},
                {'params': [self._embedding], 'lr': training_args.embedding_lr, "name": "embedding"}
            ]
            self.deformation_schedular_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.deformation_lr_max_steps)
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._intensity], 'lr': training_args.intensity_lr_init * self.spatial_lr_scale, "name": "intensity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr_init * self.spatial_lr_scale, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr_init * self.spatial_lr_scale, "name": "rotation"},
            ]
            

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    # lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.scale_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr_init,
                                            lr_final=training_args.scaling_lr_final,
                                            # lr_delay_mult=training_args.scaling_lr_delay_mult,
                                            max_steps=training_args.scaling_lr_max_steps)

        self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr_init,
                                                    lr_final=training_args.rotation_lr_final,
                                                    # lr_delay_mult=training_args.rotation_lr_delay_mult,
                                                    max_steps=training_args.rotation_lr_max_steps)

        self.intensity_scheduler_args = get_expon_lr_func(lr_init=training_args.intensity_lr_init,
                                                    lr_final=training_args.intensity_lr_final,
                                                    # lr_delay_mult=training_args.intensity_lr_delay_mult,
                                                    max_steps=training_args.intensity_lr_max_steps)


    
    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_schedular_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "intensity":
                lr = self.intensity_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "scaling":
                lr = self.scale_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group['lr'] = lr


    # 执行密集化和修剪操作
    def densify_and_prune_mix(self, max_grad, min_intensity, extent, max_screen_size, radii, bbox=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # self.tmp_radii = radii
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        ratio = (torch.norm(grads, dim = -1) >= max_grad).float().mean()
        print('[debug] densify prune: ratio = ', ratio)
        print('grad abs shape: ', grads_abs.shape)

        # Q = torch.quantile(grads_abs.reshape(-1), 1-ratio)
        if torch.any(torch.isnan(grads_abs)):
            print("NaN detected in grads_abs")
            print("NaN count in grad_abs:", torch.isnan(grads_abs).sum())
        
        if torch.all(grads_abs < 1e-6):
            print("Warning: All gradients are near zero. Skipping densify and prune operation.")
            return 0,0,0
        
        TOP_K = torch.topk(grads_abs.reshape(-1), k=int(grads_abs.numel() * ratio))
        Q = TOP_K.values[-1]
        # print('Q = ', Q)
        # print('max_grad = ', max_grad)
        
        before = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]
        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]

        # TODO: threshold selection
        prune_mask = (self.get_intensity < min_intensity).squeeze() # 创建一个掩码，标记那些密度小于指定阈值的点

        big_points_ws = self.get_scaling.max(dim=1).values > 0.05 * extent # change to 0.05 here
        prune_mask = torch.logical_or(prune_mask, big_points_ws)
        
        self.prune_points(prune_mask)
        # tmp_radii = self.tmp_radii
        # self.tmp_radii = None
        prune = self._xyz.shape[0]

        torch.cuda.empty_cache()

        return clone-before, split-clone, split-prune
    
    
    # 执行密集化和修剪操作
    def densify_and_prune(self, max_grad, min_intensity, extent, radii, max_gaussian_num, iteration, bbox=None, save_path=None, tb_writer=None):
        print('[debug] max_gaussian_num:', max_gaussian_num)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 动态调整剪枝参数
        # if iteration < 2000:
        #     max_grad = 2 * max_grad
        #     min_intensity = min_intensity * 0.8
        # elif 2000 <= iteration < 6000:
        #     max_grad = 1.5 * max_grad
        #     min_intensity = min_intensity * 0.7
        # elif 6000 <= iteration < 10000:
        #     max_grad = 1.2 * max_grad
        #     min_intensity = min_intensity * 0.6
        # elif 10000 <= iteration < 20000:
        #     min_intensity = min_intensity * 0.5
        self.tmp_radii = radii
        raw_num = self._xyz.shape[0]
        print('[debug] before densify_and_clone:', raw_num)
        
        if extent:
            
            # if not max_gaussian_num or (max_gaussian_num and grads.shape[0] < max_gaussian_num):
            self.densify_and_clone(grads, max_grad, extent, max_gaussian_num)
            after_clone = self._xyz.shape[0]
            clone_num = after_clone - raw_num
            print('[debug] clone_num:', clone_num)
            
            if after_clone >= max_gaussian_num:
                print(f'[debug] after_clone >= max_gaussian_num: {after_clone} >= {max_gaussian_num}, skip split')
                self.split_num = 0
                self.clone_num = clone_num
                return
            
            self.densify_and_split(grads, max_grad, extent, 2, max_gaussian_num)
            after_split = self._xyz.shape[0]
            split_num = after_split - after_clone
            print('[debug] split_num:', split_num)

            self.split_num = split_num
            self.clone_num = clone_num
        
        # 初始化剪枝掩码
        prune_mask = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
        
        # 低密度点移除
        low_intensity_mask = (self.get_intensity < min_intensity).squeeze()
        high_intensity_mask = (self.get_intensity > 1).squeeze()  # 高强度点
        prune_mask |= low_intensity_mask| high_intensity_mask
        # 统计低强度点数量
        num_low_intensity = low_intensity_mask.sum().item()
        num_high_intensity = high_intensity_mask.sum().item()
        # 边界外点移除
        if bbox is not None:
            xyz = self.get_xyz
            # out_of_bbox_mask = (
                # (xyz[:, 0] < bbox[0, 0]) | (xyz[:, 0] > bbox[1, 0])
                # |(xyz[:,1] < bbox[0, 1]) | (xyz[:, 1] > bbox[1, 1])
                # |(xyz[:,2] < bbox[0, 2]) | (xyz[:, 2] > bbox[1, 2])
            # )
            out_of_bbox_mask = ((xyz[:,2] < bbox[0, 2]-5) | (xyz[:, 2] > bbox[1, 2]+5))
            prune_mask = prune_mask | out_of_bbox_mask
            num_out_of_bbox = out_of_bbox_mask.sum().item()
        else:
            num_out_of_bbox = 0

        # 控制超大高斯点
        scales = self.get_scaling
        max_scale = scales.max(dim=1).values
        min_scale = scales.min(dim=1).values
        aspect_ratio = max_scale / (min_scale + 1e-6)
        
        big_point_thresh = extent * self.percent_dense * 2
        print('[debug] big_point_thresh:', big_point_thresh)
        small_point_thresh = 0.1
        big_points_ws = torch.logical_or(max_scale > big_point_thresh, aspect_ratio > 15)
        small_points_ws = min_scale < small_point_thresh

        # 计算各独立条件的剪枝数量
        prune_mask |= big_points_ws | small_points_ws
    
        # 统计输出
        num_ws = big_points_ws.sum().item()             # 新增仅世界空间剪枝
        num_small_points = small_points_ws.sum().item()     # 新增小高斯点剪枝
        num_prune_total = prune_mask.sum().item()      # 总剪枝数量
        
        print(f'[DEBUG] Prune Stats: '
            f'Total={num_prune_total} | '
            f'Low Intensity={num_low_intensity} | '
            f'High Intensity={num_high_intensity} | '
            f'Out of BBox={num_out_of_bbox} | '
            f'WS-only={num_ws} | '
            f'Small={num_small_points}')

        self.prune_points(prune_mask)
        self.visualize_anisotropy(save_path)
        self.visualize_axial_ratio(save_path)
        self.visualize_scale_distribution(save_path)
        self.visualize_intensity_distribution(save_path)
        
        tb_writer.add_scalar("Densify/Num_Cloned", self.clone_num, iteration)
        tb_writer.add_scalar("Densify/Num_Split", self.split_num, iteration)
        tb_writer.add_scalar("Prune/Total", num_prune_total, iteration)
        tb_writer.add_scalar("Prune/LowIntensity", num_low_intensity, iteration)
        tb_writer.add_scalar("Prune/HighIntensity", num_high_intensity, iteration)
        tb_writer.add_scalar("Prune/OutOfBBox", num_out_of_bbox, iteration)
        tb_writer.add_scalar("Prune/WS_only", num_ws, iteration)
        tb_writer.add_scalar("Prune/SmallPoints", num_small_points, iteration)

        torch.cuda.empty_cache()


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:# or group["name"] == "offsets":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._intensity = optimizable_tensors["intensity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_deform:
            self._embedding = optimizable_tensors["embedding"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # print('\n',group["name"])
            if len(group["params"])>1: # or group["name"]=="offsets":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:    
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # 将新的密集化点的相关特征保存在一个字典中
    def densification_postfix(self, new_xyz, new_intensity, new_scaling, new_rotation, new_embedding, new_tmp_radii=None):
        d = {"xyz": new_xyz,
             "intensity": new_intensity,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "embedding": new_embedding}
        
        # 将字典中的张量连接成可以优化的张量。这个方法的具体实现可能是将字典中的每个张量进行堆叠，以便在优化器中进行处理。
        optimizable_tensors = self.cat_tensors_to_optimizer(d) 
        # 更新模型中原始点集的相关特征，使用新的密集化后的特征
        self._xyz = optimizable_tensors["xyz"]
        self._intensity = optimizable_tensors["intensity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._embedding = optimizable_tensors["embedding"]
        # 重新初始化一些用于梯度计算和密集化操作的变量
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    # 将新的密集化点的相关特征保存在一个字典中
    def densification_postfix_nodeform(self, new_xyz, new_intensity, new_scaling, new_rotation, new_tmp_radii=None):
        d = {"xyz": new_xyz,
             "intensity": new_intensity,
             "scaling": new_scaling,
             "rotation": new_rotation}
        
        # 将字典中的张量连接成可以优化的张量。这个方法的具体实现可能是将字典中的每个张量进行堆叠，以便在优化器中进行处理。
        optimizable_tensors = self.cat_tensors_to_optimizer(d) 
        # 更新模型中原始点集的相关特征，使用新的密集化后的特征
        self._xyz = optimizable_tensors["xyz"]
        self._intensity = optimizable_tensors["intensity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))

        # 重新初始化一些用于梯度计算和密集化操作的变量
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_long_split(self, ratio_threshold=8, N=2, offset_scale=0.6, scale_factor=0.85):
        # 形状异常检测：长短轴比
        scales = self.get_scaling # [x,3]
        max_scale, max_indices = torch.max(scales, dim=1, keepdim=True)
        min_scale = torch.min(scales, dim=1, keepdim=True).values
        aspect_ratio = max_scale / (min_scale + 1e-6)

        abnormal_split_mask = aspect_ratio.squeeze() >= ratio_threshold
        num_abnormal = abnormal_split_mask.sum().item()
        print(f"Splitting {num_abnormal} Gaussians due to abnormal aspect ratio")
        
        if num_abnormal == 0:
            return
        
        base_xyz = self.get_xyz[abnormal_split_mask]
        base_scales = scales[abnormal_split_mask]
        base_rots = self._rotation[abnormal_split_mask]
        base_intensities = self.get_intensity[abnormal_split_mask]

         # 使用 gather 解决索引问题
        max_axis_dir = torch.zeros_like(base_scales)
        max_axis_dir.scatter_(1, max_indices[abnormal_split_mask], 1)
        max_axis_dir = torch.bmm(build_rotation(base_rots), max_axis_dir.unsqueeze(-1)).squeeze(-1)
        max_axis_dir = F.normalize(max_axis_dir, dim=-1)

        offset = max_axis_dir * (base_scales.gather(1, max_indices[abnormal_split_mask]) * offset_scale)

        # 生成扰动方向
        random_vec = torch.randn_like(max_axis_dir)
        random_vec = F.normalize(random_vec, dim=-1)

        # 正交化处理：生成与最大轴正交的方向
        perturbation_vec = random_vec - torch.sum(random_vec * max_axis_dir, dim=-1, keepdim=True) * max_axis_dir
        perturbation_vec = F.normalize(perturbation_vec, dim=-1) * (0.1 * offset.norm(dim=-1, keepdim=True))

        # 分裂点位置计算
        new_xyz_1 = base_xyz + offset + perturbation_vec
        new_xyz_2 = base_xyz - offset - perturbation_vec
        new_xyz = torch.cat((new_xyz_1, new_xyz_2), dim=0)

        new_scales = base_scales.repeat(N, 1) / (N ** (1/3)) * scale_factor
        new_rots = base_rots.repeat(N, 1)
        new_intensity = base_intensities.repeat(N)

        if self.use_deform:
            new_embedding = self._embedding[abnormal_split_mask].repeat(N,1)
            self.densification_postfix(new_xyz, new_intensity, new_scales, new_rots, new_embedding)
        else:
            self.densification_postfix_nodeform(new_xyz, new_intensity, new_scales, new_rots)

        torch.cuda.empty_cache()

        # 创建一个修剪（pruning）的过滤器，将新生成的点添加到原始点的掩码之后。根据修剪过滤器，修剪模型中的一些参数
        prune_filter = torch.cat((abnormal_split_mask, torch.zeros(N * num_abnormal, device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        print("Split long ellipsoids completed.")


    # def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
    #     n_init_points = self.get_xyz.shape[0]

    #     # 基于梯度选择分裂点
    #     padded_grad = torch.zeros((n_init_points), device="cuda") 
    #     padded_grad[:grads.shape[0]] = grads.squeeze()
    #     selected_pts_mask = padded_grad >= grad_threshold

    #     scales = self.get_scaling
    #     # 输出轴长分布
    #     aspect_ratio = scales[:,2] / scales[:,0]
        
    #     # 过滤掉那些缩放scaling大于一定百分比的场景范围的点
    #     selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(scales, dim=1).values > self.percent_dense * scene_extent)
    #     print(f"Selected points for split: {selected_pts_mask.sum().item()}")
        
    #     selected_xyz = self.get_xyz[selected_pts_mask]
    #     selected_intensity = self.get_intensity[selected_pts_mask]
    #     selected_scales = self.get_scaling[selected_pts_mask]
    #     selected_rots = self._rotation[selected_pts_mask]
        
    #     # 找出每个点的最大轴索引（0, 1, or 2）
    #     max_dim = torch.argmax(selected_scales, dim=1)  # [M]

    #     # 按照高斯缩放生成扰动方向
    #     stds = selected_scales.repeat(N,1)
    #     means = torch.zeros((stds.size(0), 3), device="cuda")
    #     samples = torch.normal(mean=means, std=stds) # 使用均值和标准差生成样本

    #     # 单位化
    #     samples /= samples.norm(dim=-1, keepdim=True) + 1e-6

    #     adaptive_scaling = torch.clamp(1 / aspect_ratio[selected_pts_mask].unsqueeze(-1), 0.5, 1.5)
    #     print(f"Adaptive scaling: {adaptive_scaling.mean():.3f}")
    #     adaptive_scaling = adaptive_scaling.repeat(N, 1)  # 让它与 samples 形状匹配
    #     samples *= adaptive_scaling
        
        
    #     # 添加随机旋转扰动
    #     random_rot = torch.rand((samples.shape[0], 3), device="cuda") * 2 * torch.pi
    #     cos_r = torch.cos(random_rot)
    #     sin_r = torch.sin(random_rot)
    #     random_rotation = torch.stack([
    #         cos_r[:, 0], -sin_r[:, 0], torch.zeros_like(cos_r[:, 0]),
    #         sin_r[:, 0], cos_r[:, 0], torch.zeros_like(cos_r[:, 0]),
    #         torch.zeros_like(cos_r[:, 0]), torch.zeros_like(cos_r[:, 0]), torch.ones_like(cos_r[:, 0])
    #     ], dim=1).reshape(-1, 3, 3)
    #     samples = torch.bmm(random_rotation, samples.unsqueeze(-1)).squeeze(-1)

    #     rots = build_rotation(selected_rots).repeat(N,1,1) # 为每个point构建旋转矩阵，并将其重复N次
        
    #     # 新的协方差矩阵加上原本高斯点位置，也就是将旋转后的样本点添加到原始点的位置
    #     new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N,1) 
    #     new_scaling = self.scaling_inverse_activation(selected_scales.repeat(N,1)/(0.6*N)) # 生成新的缩放参数 /1.2

    #     new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
    #     new_intensity = self._intensity[selected_pts_mask].repeat(N)
    #     print('new scaling: ', new_scaling.max(), new_scaling.min())
    #     if self.use_deform:
    #         new_embedding = self._embedding[selected_pts_mask].repeat(N,1)
    #         self.densification_postfix(new_xyz, new_intensity, new_scaling, new_rotation, new_embedding)
    #     else:
    #         self.densification_postfix_nodeform(new_xyz, new_intensity, new_scaling, new_rotation)

    #     torch.cuda.empty_cache()
 
    #     prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
    #     self.prune_points(prune_filter)

    #     print("Densify and split completed.")



    # def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
    #     n_init_points = self.get_xyz.shape[0]
    #     # Extract points that satisfy the gradient condition
    #     # 创建一个长度为初始点数量的梯度张量，并将计算得到的梯度填充进去
    #     padded_grad = torch.zeros((n_init_points), device="cuda") 
    #     padded_grad[:grads.shape[0]] = grads.squeeze()
        
    #     # 梯度分裂：选择满足梯度条件的点
    #     selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
    #     # 过滤掉那些缩放scaling大于一定百分比的场景范围的点
    #     scales = self.get_scaling
    #     # 输出轴长分布
    #     z_over_x = scales[:,2] / scales[:,0]
    #     print(f"Z/X轴长比 - 均值: {z_over_x.mean():.3f}, 方差: {z_over_x.var():.3f}")

    #     selected_pts_mask = torch.logical_and(selected_pts_mask,
    #                                           torch.max(scales, dim=1).values > self.percent_dense * scene_extent)

    #     stds = self.get_scaling[selected_pts_mask].repeat(N,1)
    #     means = torch.zeros((stds.size(0), 3), device="cuda")
    #     samples = torch.normal(mean=means, std=stds) # 使用均值和标准差生成样本
    #     rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1) # 为每个point构建旋转矩阵，并将其重复N次
        
    #     # 新的协方差矩阵加上原本高斯点位置，也就是将旋转后的样本点添加到原始点的位置
    #     new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N,1) 
    #     new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1)/(0.8*N)) # 生成新的缩放参数 /1.6
    #     new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
    #     new_intensity = self.intensity_inverse_activation(self.get_intensity[selected_pts_mask].repeat(N))
    #     # new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        
    #     if self.use_deform:
    #         new_embedding = self._embedding[selected_pts_mask].repeat(N,1)
    #         # 对新生成的点执行后处理操作
    #         self.densification_postfix(new_xyz, new_intensity, new_scaling, new_rotation, new_embedding)
    #     else:
    #         self.densification_postfix_nodeform(new_xyz, new_intensity, new_scaling, new_rotation)
            
    #     # 创建一个修剪（pruning）的过滤器，将新生成的点添加到原始点的掩码之后。根据修剪过滤器，修剪模型中的一些参数
    #     prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
    #     self.prune_points(prune_filter)

    #     print("Densify and split completed.")

    

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, max_gaussian_num=5000000):
        n_init_points = self.get_xyz.shape[0]
        if n_init_points >= max_gaussian_num:
            print('[debug] Max gaussian number reached, skipping split...')
            return
        
        # 基于梯度选择分裂点
        padded_grad = torch.zeros((n_init_points), device="cuda") 
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
      
        print(f"Selected points for split: {selected_pts_mask.sum().item()}")
        if selected_pts_mask.sum().item() < 1:
            print('[debug] no points to be split...')
            return

        # 计算分裂后将新增的高斯数
        num_new_gaussians = selected_pts_mask.sum().item()
        if n_init_points + num_new_gaussians > max_gaussian_num:
            remaining_capacity = max_gaussian_num - n_init_points
            max_points_to_split = remaining_capacity
            if max_points_to_split <= 0:
                print('[debug] Not enough capacity for even one split operation...')
                return
            
            # 选择梯度最大的max_points_to_split个点进行分裂
            grad_values = padded_grad[selected_pts_mask]
            _, sorted_indices = torch.topk(grad_values, k=max_points_to_split)
            original_indices = torch.where(selected_pts_mask)[0]
            selected_pts_mask = torch.zeros_like(selected_pts_mask)
            selected_pts_mask[original_indices[sorted_indices]] = True
            print(f'Adjusted split points to {max_points_to_split} due to max_gaussian_num limit')

        selected_xyz = self.get_xyz[selected_pts_mask]
        selected_intensity = self.get_intensity[selected_pts_mask]
        selected_scales = self.get_scaling[selected_pts_mask]  # [M, 3]
        selected_rots = self.get_rotation[selected_pts_mask]      # [M, 4]

        # 构建旋转矩阵 [M, 3, 3]
        rots = build_rotation(selected_rots)

        # 找出每个点的最大轴索引（0, 1, or 2）
        max_dim = torch.argmax(selected_scales, dim=1)  # [M]

        # 构造本地长轴方向单位向量
        e = torch.zeros((selected_scales.shape[0], 3), device="cuda")  # [M, 3]
        e.scatter_(1, max_dim.unsqueeze(1), 1.0)  # 在每一行中，将最大轴位置置为1

        # 旋转到世界坐标系 -> [M, 3, 1] = [M, 3, 3] @ [M, 3, 1]
        long_axis_world = torch.bmm(rots, e.unsqueeze(-1)).squeeze(-1)  # [M, 3]

        # 沿着长轴方向生成正负扰动，repeat N 次
        long_axis_world = long_axis_world.repeat_interleave(N, dim=0)  # [M*N, 3]
        scales_along_long_axis = torch.gather(
            selected_scales, 1, max_dim.unsqueeze(1)
        ).squeeze(1).repeat_interleave(N)  # [M*N]

        perturbation = long_axis_world * (scales_along_long_axis / (2 * N)).unsqueeze(1)  # [M*N, 3]

        # 交替正负方向
        signs = torch.ones_like(perturbation)
        signs[::2] *= -1
        perturbation *= signs

        # 新点位置
        new_xyz = selected_xyz.repeat_interleave(N, dim=0) + perturbation

        # repeated_scales = selected_scales.repeat_interleave(N, dim=0)  # [M*N, 3]
        # repeated_max_dim = max_dim.repeat_interleave(N)  # [M*N]

        # idx = torch.arange(repeated_scales.shape[0], device="cuda")
        # shrinked_scales = repeated_scales.clone()
        # shrinked_scales[idx, repeated_max_dim] /= (0.8 * N)
        # new_scaling = self.scaling_inverse_activation(shrinked_scales)

        # 新的 scale、rotation、intensity
        new_scaling = self.scaling_inverse_activation(selected_scales.repeat_interleave(N, dim=0) / (0.8 * N))
        new_rotation = selected_rots.repeat_interleave(N, dim=0)
        new_intensity = selected_intensity.repeat_interleave(N) * 0.8

        print('new scaling: ', new_scaling.max(), new_scaling.min())

        if self.use_deform:
            new_embedding = self._embedding[selected_pts_mask].repeat_interleave(N, dim=0)
            self.densification_postfix(new_xyz, new_intensity, new_scaling, new_rotation, new_embedding)
        else:
            self.densification_postfix_nodeform(new_xyz, new_intensity, new_scaling, new_rotation)

        torch.cuda.empty_cache()

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=torch.bool)))
        self.prune_points(prune_filter)
        print("Densify and split completed.")
        

    def only_prune_intensity(self, min_intensity):
        prune_mask = (self.get_intensity < min_intensity).squeeze()
        self.prune_points(prune_mask)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()
    
    def only_prune(self, min_intensity, extent, bbox=None, max_num=5000000):
        # 初始化剪枝掩码
        prune_mask = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
        
        # 移除低强度点
        low_intensity_mask = (self.get_intensity < min_intensity).squeeze()
        prune_mask |= low_intensity_mask
        num_low_intensity = low_intensity_mask.sum().item()

        # 边界外点移除
        if bbox is not None:
            xyz = self.get_xyz
            # out_of_bbox_mask = (
                # (xyz[:, 0] < bbox[0, 0]) | (xyz[:, 0] > bbox[1, 0])
                # |(xyz[:,1] < bbox[0, 1]) | (xyz[:, 1] > bbox[1, 1])
                # |(xyz[:,2] < bbox[0, 2]) | (xyz[:, 2] > bbox[1, 2])
            # )
            out_of_bbox_mask = ((xyz[:,2] < bbox[0, 2]) | (xyz[:, 2] > bbox[1, 2]))
            prune_mask |= out_of_bbox_mask
            num_out_of_bbox = out_of_bbox_mask.sum().item()
        else:
            num_out_of_bbox = 0
        
        # 移除异常尺寸和比例的点
        scales = self.get_scaling
        max_scale = scales.max(dim=1).values
        min_scale = scales.min(dim=1).values
        aspect_ratio = max_scale / (min_scale + 1e-6)
        
        big_point_thresh = 0.05 * extent
        small_point_thresh = 0.05
        big_points_ws = torch.logical_or(max_scale > big_point_thresh, aspect_ratio > 15)
        small_points_ws = min_scale < small_point_thresh
        
        prune_mask |= big_points_ws | small_points_ws
        num_big_points = big_points_ws.sum().item()
        num_small_points = small_points_ws.sum().item()     # 新增小高斯点剪枝
        num_prune_total = prune_mask.sum().item()      # 总剪枝数量
        
        print(f'[DEBUG] Only Prune Stats: '
            f'Total={num_prune_total} | '
            f'Low Intensity={num_low_intensity} | '
            f'Out of BBox={num_out_of_bbox} | '
            f'WS={num_big_points} | '
            f'Small={num_small_points}')

        self.prune_points(prune_mask)
        
        # remaining = self.get_xyz.shape[0]
        # if remaining > max_num:
        #     print(f'[DEBUG] Remaining points exceed max_num ({max_num}), pruning further...')
        #     intensity = self.get_intensity.squeeze()
        #     _, keep_indices = torch.topk(intensity, max_num, largest=True)
        #     keep_mask = torch.zeros_like(intensity, dtype=torch.bool)
        #     keep_mask[keep_indices] = True
        #     prune_mask = ~keep_mask
        #     self.prune_points(prune_mask)
        #     print(f'[DEBUG] Remaining points after pruning: {self.get_xyz.shape[0]}')

        self.visualize_anisotropy()
        self.visualize_axial_ratio()
        self.visualize_scale_distribution()
        self.visualize_intensity_distribution()
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()
        

    def densify_and_clone_mix(self, grads, grad_threshold, grad_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # 对于每个点计算梯度的L2范数，如果≥指定的阈值则标为true
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grad_abs, dim=-1) >= grad_abs_threshold, True, False)
        # 计算 mask 中为 True 的数量
        num_true = torch.sum(selected_pts_mask).item()
        print("Number of True values in selected_pts_mask:", num_true)
        print("Number of True values in selected_pts_mask_abs:", torch.sum(selected_pts_mask_abs).item())

        # 在上述掩码的基础上进一步过滤掉那些缩放大于一定百分比的场景范围的点。这样可以确保新添加的点不会太远离原始数据。
        # selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
       
        # 根据掩码选取符合条件的点的其他特征。
        new_xyz = self._xyz[selected_pts_mask]
        new_intensity = self._intensity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self.use_deform:
            new_embedding = self._embedding[selected_pts_mask]
            self.densification_postfix(new_xyz, new_intensity, new_scaling, new_rotation, new_embedding)
        else:
            self.densification_postfix_nodeform(new_xyz, new_intensity, new_scaling, new_rotation)
            
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent, max_gaussian_num):
        # 增加各向异性梯度阈值
        # adjusted_threshold = torch.where(
        #     (self.get_scaling[:,2] / self.get_scaling[:,0]) > 2,
        #     1.5 * grad_threshold, 
        #     grad_threshold
        # )

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        # 在上述掩码的基础上进一步过滤掉那些缩放大于一定百分比的场景范围的点。这样可以确保新添加的点不会太远离原始数据。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        num_new_gaussians = selected_pts_mask.sum().item()
        current_num_gaussians = self.get_xyz.shape[0]
        if current_num_gaussians + num_new_gaussians > max_gaussian_num:
            remaining_slots = max_gaussian_num - current_num_gaussians
            if remaining_slots <= 0:
                print(f"Current number of Gaussians ({current_num_gaussians}) exceeds max limit ({max_gaussian_num}). No new Gaussians added.")
                return
            # Select only the top 'remaining_capacity' points with highest gradients
            grad_norms = torch.norm(grads, dim=-1)
            grad_values = grad_norms[selected_pts_mask]
            # Get indices of the points with highest gradients
            _, sorted_indices = torch.topk(grad_values, k=remaining_slots)
            # Create a new mask that only includes these top points
            original_indices = torch.where(selected_pts_mask)[0]
            selected_pts_mask = torch.zeros_like(selected_pts_mask)
            selected_pts_mask[original_indices[sorted_indices]] = True


        # 根据掩码选取符合条件的点的其他特征。
        new_xyz = self._xyz[selected_pts_mask]
        new_intensity = self._intensity[selected_pts_mask] * 0.5
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]
        self._intensity[selected_pts_mask] = new_intensity

        if self.use_deform:
            new_embedding = self._embedding[selected_pts_mask]
            self.densification_postfix(new_xyz, new_intensity, new_scaling, new_rotation, new_embedding)
        else:
            self.densification_postfix_nodeform(new_xyz, new_intensity, new_scaling, new_rotation)
            
    

    def add_densification_stat(self, viewspace_point_tensor, update_filter):
        if viewspace_point_tensor.grad is None:
            print("No gradients available for viewspace_point_tensor")

        update_filter = update_filter.to(self.xyz_gradient_accum.device)
        xyz =  self.xyz_gradient_accum[update_filter]
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] += torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1


    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.name_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    print(name, ":", weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name, ":", weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)


    def visualize_gaussians(self, points, intensities, scales, volume_x:int, volume_y:int, volume_z:int):
        # 创建一个新的图形窗口
        fig = mlab.figure(bgcolor=(0, 0, 0))
        print('points.shape: ', points.shape)
        print('intensities.shape: ', intensities.shape)
        print('intensity min, max: ', intensities.min(), intensities.max())
        print('scale.shape: ', scales.shape)
        print('point x:', points[:,0].min(), points[:,0].max())
        print('point y:', points[:,1].min(), points[:,1].max())
        print('point z:', points[:,2].min(), points[:,2].max())

        
        # 使用 scalar_scatter 创建点，设置强度作为点的颜色映射值
        scatter = mlab.pipeline.scalar_scatter(points[:, 0], points[:, 1], points[:, 2], intensities)
        scatter.mlab_source.dataset.point_data.add_array(scales[:, 0])
        scatter.mlab_source.dataset.point_data.get_array(1).name = 'scales'
        scatter.update()

        # 可视化每个点为一个球体
        glyphs = mlab.pipeline.glyph(scatter)
        glyphs.glyph.glyph_source.glyph_source = glyphs.glyph.glyph_source.glyph_dict['sphere_source']  # 设置为球体

        # 将缩放模式设置为使用 'vector' 并指定每个点的大小
        glyphs.glyph.scale_mode = 'scale_by_vector'
        glyphs.glyph.glyph.scale_factor = 1.0  # 控制总体缩放因子，根据需要调整
        glyphs.mlab_source.dataset.point_data.vectors = np.column_stack((scales[:, 0], scales[:, 0], scales[:, 0]))  # 使用 scales 控制球体大小

        # 设置颜色映射
        glyphs.module_manager.scalar_lut_manager.lut_mode = 'jet'

        # 设置坐标轴和轮廓线
        mlab.outline()
        mlab.axes(extent=(-volume_x/2,volume_x/2,-volume_y/2,volume_y/2,-volume_z/2,volume_z/2))
        mlab.show()


    def visualize_anisotropy(self, path=None):
        """可视化各向异性分布"""
        scales = self.get_scaling
        scales = scales.detach().cpu().numpy()
        aspect_ratios = np.max(scales, axis=1) / np.min(scales, axis=1)
        
        plt.figure(figsize=(10,6))
        plt.hist(aspect_ratios, bins=50, range=(1,20))
        plt.axvline(5, color='r', linestyle='--')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Count')
        plt.title('Gaussian Anisotropy Distribution')
        plt.savefig(os.path.join(path, 'anisotropy_distribution.png') if path else 'anisotropy_distribution.png')
        plt.close()


    def visualize_axial_ratio(self, path=None):
        """监控各轴向比例变化"""
        scales = self.get_scaling
        scales = scales.detach().cpu().numpy()
        ratios = {
            'Z/X': scales[:,2]/scales[:,0],
            'Z/Y': scales[:,2]/scales[:,1],
            'X/Y': scales[:,0]/scales[:,1]
        }
        
        fig = plt.figure(figsize=(12,6))
        for i, (name, values) in enumerate(ratios.items()):
            plt.subplot(2,2,i+1)
            plt.hist(values, bins=50, range=(0,10))
            plt.title(f'{name} Ratio Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, "axial_ratios.png") if path else 'axial_ratios.png')
        plt.close(fig)

    def visualize_scale_distribution(self, path=None):
        """可视化尺度分布"""
        scales = self.get_scaling
        scales = scales.detach()
        max_scale = torch.max(scales, dim=1).values.cpu().numpy()
        min_scale = torch.min(scales, dim=1).values.cpu().numpy()
        mean_scale = torch.mean(scales, dim=1).cpu().numpy()
        aspect_ratio = max_scale / (min_scale + 1e-6)

        fig = plt.figure(figsize=(15, 10))

        # 1. 绘制最大尺度分布
        plt.subplot(2, 2, 1)
        sns.histplot(max_scale, bins=50, color='blue', kde=True)
        plt.title('Max Scale Distribution')

        # 2. 绘制最小尺度分布
        plt.subplot(2, 2, 2)
        sns.histplot(min_scale, bins=50, color='green', kde=True)
        plt.title('Min Scale Distribution')

        # 3. 绘制长短轴比值
        plt.subplot(2, 2, 3)
        sns.histplot(aspect_ratio, bins=50, color='red', kde=True)
        plt.title('Aspect Ratio Distribution (Max/Min)')

        # 4. 长短轴比 vs 最大尺度散点图
        plt.subplot(2, 2, 4)
        plt.scatter(max_scale, aspect_ratio, alpha=0.5, color='purple')
        plt.xlabel('Max Scale')
        plt.ylabel('Aspect Ratio')
        plt.title('Aspect Ratio vs Max Scale')

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'scale_distribution.png') if path else 'scale_distribution.png')
        plt.close(fig)


    def visualize_intensity_distribution(self, path=None):
        """可视化intensity分布及其与缩放的关系"""        
        intensities = self.get_intensity.detach().cpu().numpy().flatten()
        scales = self.get_scaling.detach().cpu().numpy()
        max_scales = np.max(scales, axis=1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 直方图
        ax1.hist(intensities, bins=100, range=(0, 1), alpha=0.7, color='blue')
        ax1.set_xlabel('Intensity Value')
        ax1.set_ylabel('Count')
        ax1.set_title('Intensity Distribution Histogram')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 散点图：intensity vs 最大缩放
        ax2.scatter(intensities, max_scales, s=1, alpha=0.5, c='red')
        ax2.set_xlabel('Intensity')
        ax2.set_ylabel('Max Scale')
        ax2.set_title('Intensity vs Scale Relationship')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, "intensity_dist.png") if path else 'intensity_dist.png')
        plt.close(fig)







def test_visualize():
    print('debug...')
    num_points = 200
    min_coord = [0, 0, 0]
    max_coord = [540, 540, 270]
    your_points = np.random.rand(num_points, 3) * (np.array(max_coord) - np.array(min_coord)) + np.array(min_coord)
    your_intensities = np.random.rand(num_points, 1)
    reconstruction = BasicPointCloud(points=your_points, intensities=your_intensities)

    model = GaussianModel_cryoET(disable_xyz_log_activation=True)
    model.create_from_wbp(reconstruction, intensity_threshold=0.1, init_size=3000, spatial_lr_scale=1.0, image_height=1024, image_width=1024, grid_size=8, volume_x=540, volume_y=540, volume_z=270, factor=1.0)
    
    points = model._xyz.detach().cpu().numpy()
    intensities = model._intensity.detach().squeeze().cpu().numpy()
    scales = model._scaling.detach().cpu().numpy()
    print('visualize...')
    # model.visualize_gaussians(points, intensities, scales)




if __name__ == "__main__":
    test_visualize()