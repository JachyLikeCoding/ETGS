import torch
import numpy as np
from torch import nn
import os, sys
from mayavi import mlab
import matplotlib.pyplot as plt
from utils.system_utils import mkdir_p
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import inverse_sigmoid, build_rotation, build_scaling_rotation, strip_symmetric, get_expon_lr_func
from simple_knn._C import distCUDA2
from plyfile import PlyData, PlyElement

def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def inverse_log_transform(y):
    return torch.sign(y) * (torch.expm1(torch.abs(y)))


class GaussianModel_cryoET:
    '''设置各种激活和变换函数'''
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
        # self.scaling_activation = torch.exp # 用指数函数将尺度限制为非负数
        # self.scaling_inverse_activation = torch.log # 尺度参数的逆激活函数，用于梯度回传
        self.scaling_activation = torch.nn.functional.softplus  # 使用softplus函数限制尺度为正数，但增长更平缓
        self.scaling_inverse_activation = lambda x: torch.log(torch.exp(x) - 1)  # 对应softplus的逆函数

        self.covariance_activation = build_covariance_from_scaling_rotation # 协方差矩阵的激活函数
        
        self.intensity_activation = torch.sigmoid # 将强度限制在0-1范围
        self.intensity_inverse_activation = inverse_sigmoid # 强度的逆激活函数
        
        self.rotation_activation = torch.nn.functional.normalize #用于归一化旋转矩阵的函数

    
    def __init__(self, disable_xyz_log_activation=True):
        self.disable_xyz_log_activation = disable_xyz_log_activation
        self._xyz = torch.empty(0) # 3D高斯的中心位置（均值）
        self._intensity = torch.empty(0) # 强度（密度）   
        self._scaling = torch.empty(0)  # 尺度参数
        self._rotation = torch.empty(0) # 旋转参数，用四元数表示
        self.max_radii2D = torch.empty(0)   # 投影到2D时，每个2D gaussian最大的半径
        self.xyz_gradient_accum = torch.empty(0)  # 3D gaussian中心位置的累计梯度
        self.denom = torch.empty(0)
        self.optimizer = None   # 优化器，用于调整上述参数以改进模型
        self.percent_dense = 0
        self.spatial_lr_scale = 1.0
        self.setup_functions()


    def capture(self):
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
        )


    def restore(self, model_args, training_args):
        (self._xyz,
        self._intensity,
        self._scaling,
        self._rotation,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
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
        # return self._rotationf
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_intensity(self):
        # return self._intensity
        return self.intensity_activation(self._intensity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def reset_intensity(self):
        intensities_new = inverse_sigmoid(torch.min(self.get_intensity, torch.ones_like(self.get_intensity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(intensities_new, "intensity")
        self._intensity = optimizable_tensors["intensity"]


    # 获得初始点
    def create_from_fbp(self, fbp: BasicPointCloud, intensity_threshold:float, init_size:int, spatial_lr_scale:float, image_width:int, image_height:int, grid_size:int,
                        volume_x:int, volume_y:int, volume_z:int):
        """
        从粗重建数据初始化模型参数
        :param fbp: FBP粗重建数据, 包含点的位置和密度值
        :param spatial_lr_scale: 空间学习率缩放因子，影响位置参数的学习率
        """
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(np.asarray(fbp.points)).float().cuda() # (P, 3)

        # Calculate max and min values along each dimension
        max_values, _ = torch.max(points, dim=0)
        min_values, _ = torch.min(points, dim=0)
        max_x, max_y, max_z = max_values
        min_x, min_y, min_z = min_values
        print(f"\nMax values:  x: {max_x.item()}, y: {max_y.item()}, z: {max_z.item()}")
        print(f"\nMin values:  x: {min_x.item()}, y: {min_y.item()}, z: {min_z.item()}")

        intensities = torch.tensor(np.asarray(fbp.intensities)).float().cuda()
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())

        # intensities = inverse_sigmoid(intensities) # (P, 1)
        print('\n[debug]: intensities.shape:', intensities.shape)
        print("\nNumber of points at initialisation : ", points.shape[0])
        print('intensities max:', max(intensities), '| min:', min(intensities))

        # Filter points based on intensity threshold
        tmp = intensities
        high_intensity_mask = tmp.squeeze() >  intensity_threshold
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
        sampled_intensities = inverse_sigmoid(high_intensity_values[selected_indices])
        print("Number of points after grid select: ", sampled_points.shape[0])

        scale_x, scale_y = image_width/volume_x, image_height/volume_y
        scale_z = scale_y

        center = torch.tensor([volume_x * 0.5, volume_y * 0.5, volume_z * 0.5], device="cuda")
        sampled_points -= center
        sampled_points[:, 0] *= scale_x
        sampled_points[:, 1] *= scale_y
        sampled_points[:, 2] *= scale_z
        
        # 计算每个点到其最近的K个点的平均距离的平方，用于确定高斯的尺度参数
        dist2 = torch.clamp_min(distCUDA2(sampled_points), 1e-6) # (P, )
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) # (P, 3)

        # 初始化每个点的旋转参数为单元四元数（无旋转）
        rots = torch.zeros((sampled_points.shape[0], 4), device="cuda") # (P, 4)
        rots[:, 0] = 1 # 四元数的实部为1，表示无旋转
        print('\n[debug]: sampled_intensities.shape:', sampled_intensities.shape)

        # 将以上计算的参数设置为模型的可训练参数
        self._xyz = nn.Parameter(sampled_points.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._intensity = nn.Parameter(sampled_intensities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 存储2D投影的最大半径，初始化为0


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'intensity']
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    

    def save_gaussianpoints(self, path):
        mkdir_p(os.path.dirname(path))

        # 将数据转为 numpy 数组并四舍五入保留 4 位小数
        xyz = np.round(self._xyz.detach().cpu().numpy(), 4)
        normals = np.zeros_like(xyz)
        intensities = np.round(self._intensity.detach().cpu().numpy().reshape(-1, 1), 4)
        scale = np.round(self._scaling.detach().cpu().numpy(), 4)
        rotation = np.round(self._rotation.detach().cpu().numpy(), 4)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        print('\n[debug] xyz.shape', xyz.shape)
        print('[debug] intensities.shape', intensities.shape)
        print('[debug] scale.shape', scale.shape)
        print('[debug] rotation.shape', rotation.shape)
        
        attributes = np.concatenate((xyz, normals, intensities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def save_gaussianpoints_rgb(self, path):
        mkdir_p(os.path.dirname(path))

        # 将数据转为 numpy 数组并四舍五入保留 4 位小数
        xyz = np.round(self._xyz.detach().cpu().numpy(), 4)
        normals = np.zeros_like(xyz)  # 法线暂时不使用
        intensities = np.round(self._intensity.detach().cpu().numpy().reshape(-1, 1), 4)
        scale = np.round(self._scaling.detach().cpu().numpy(), 4)
        rotation = np.round(self._rotation.detach().cpu().numpy(), 4)

        # 将 intensity 归一化到 [0, 1] 范围，并映射为 RGB
        rgb_colors = self.intensity_to_rgb(intensities)
        # 计算不透明度（可以根据强度计算不透明度，或者根据缩放等）
        opacity = self.intensity_to_opacity(intensities)

        # 构建 PLY 文件的字段名
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('opacity', 'f4')]

        # 添加 scale 和 rotation 信息
        for i in range(scale.shape[1]):  # 三维的 scale 属性
            dtype_full.append(('scale_{}'.format(i), 'f4'))
        for i in range(rotation.shape[1]):  # 四元数的 rotation 属性
            dtype_full.append(('rot_{}'.format(i), 'f4'))

        # 创建元素存储点的属性
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # 填充数据
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]

        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]

        elements['red'] = rgb_colors[:, 0]
        elements['green'] = rgb_colors[:, 1]
        elements['blue'] = rgb_colors[:, 2]
        elements['opacity'] = opacity  # 将不透明度填充到对应位置
        
        # 填充缩放因子 (scale)，每个点有三个值
        for i in range(scale.shape[1]):
            elements['scale_{}'.format(i)] = scale[:, i]

        # 填充旋转矩阵 (rotation)，每个点有四个值
        for i in range(rotation.shape[1]):
            elements['rot_{}'.format(i)] = rotation[:, i]

        # 创建 PLY 元素
        el = PlyElement.describe(elements, 'vertex')

        # 保存为 PLY 文件
        PlyData([el]).write(path)



    def intensity_to_rgb(self, intensity):
        print(f'[debug] intensity max: {np.max(intensity)}, min: {np.min(intensity)}')
        """ 将密度值归一化并映射到 RGB 色调 """
        # 归一化 intensity 到 [0, 1] 范围
        intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        intensity = np.clip(intensity, 0, 1)
        # 将 intensity 映射到 [0, 255] 范围的灰度色调
        rgb = (intensity * 255).astype(np.uint8)
        
        # 返回 RGB 值，每个点将有对应的红、绿、蓝通道值
        return np.repeat(rgb, 3, axis=1).reshape(-1, 3)  # 返回每个点的 RGB 三个通道


    def intensity_to_opacity(self, intensity):
        """ 根据强度计算不透明度，强度越大，不透明度越高 """
        # 归一化 intensity 到 [0, 1] 范围
        intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        intensity = np.clip(intensity, 0, 1)
        # 假设强度直接映射到不透明度
        opacity = intensity.flatten()  # 这里可以根据需要做进一步的调整

        return opacity


    def save_csv(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        intensities = self._intensity.detach().cpu().numpy().reshape(-1,1)
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        attributes = np.concatenate((xyz, normals, intensities, scale, rotation), axis=1)
        
        header = "x,y,z,nx,ny,nz,intensity,scale,rotation"
        np.savetxt(path, attributes, delimiter=',', header=header, comments='')


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

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._intensity = nn.Parameter(torch.tensor(intensities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        print('********* [debug]: load ply success...')

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
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._intensity], 'lr': training_args.intensity_lr, "name": "intensity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 创建学习率调度器，用于对中心点位置的学习率进行调整
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        # 对密度值的学习率进行调整
        # self.intensity_scheduler_args = get_expon_lr_func(lr_init=training_args.intensity_lr_init,
        #                                             lr_final=training_args.intensity_lr_final,
        #                                             lr_delay_mult=training_args.intensity_lr_delay_mult,
        #                                             max_steps=training_args.intensity_lr_max_steps)

    
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            # if param_group["name"] == "intensity":
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr


    # 执行密集化和修剪操作
    def densify_and_prune(self, max_grad, min_intensity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom # 计算密度估计的梯度
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim = -1) >= max_grad).float().mean()
        print('[debug] densify prune: ratio = ', ratio)
        Q = torch.quantile(grads_abs.reshape(-1), 1-ratio)
        print('Q = ', Q)
        print('max_grad = ', max_grad)
        
        before = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]
        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]

        # TODO: threshold selection
        prune_mask = (self.get_intensity < 0.02).squeeze() # 创建一个掩码，标记那些密度小于指定阈值的点

        # # 过滤掉那些不在x, y, z范围内的点
        # in_x_range = (self.get_xyz[:, 0] >= -800) & (self.get_xyz[:, 0] <= 800)
        # in_y_range = (self.get_xyz[:, 1] >= -800) & (self.get_xyz[:, 1] <= 800)
        # in_z_range = (self.get_xyz[:, 2] >= -300) & (self.get_xyz[:, 2] <= 300)

        # # 将范围检查的结果合并
        # in_range_mask = in_x_range & in_y_range & in_z_range
        # prune_mask = torch.logical_or(prune_mask, ~in_range_mask)  # 合并之前的掩码和范围过滤条件

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.05 * extent # change to 0.05 here
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        prune = self._xyz.shape[0]

        torch.cuda.empty_cache()

        return clone-before, split-clone, split-prune
    

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
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

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            print('\n',group["name"])
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if stored_state is not None:
                print("[debug] stored_state['exp_avg'] shape:", stored_state["exp_avg"].shape)
                print("[debug] extension_tensor shape:", extension_tensor.shape)
                print("[debug] stored_state['exp_avg_sq'] shape:", stored_state["exp_avg_sq"].shape)

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
    def densification_postfix(self, new_xyz, new_intensity, new_scaling, new_rotation):
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

        # 重新初始化一些用于梯度计算和密集化操作的变量
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")



    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # 创建一个长度为初始点数量的梯度张量，并将计算得到的梯度填充进去
        padded_grad = torch.zeros((n_init_points), device="cuda") 
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # 选择满足梯度条件的点
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()

        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        # 过滤掉那些缩放scaling大于一定百分比的场景范围的点
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        
        # 为每个点生成新的样本，其中 stds 是点的缩放， means是均值
        # stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.clamp(self.get_scaling[selected_pts_mask].repeat(N,1), min=0)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds) # 使用均值和标准差生成样本
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1) # 为每个point构建旋转矩阵，并将其重复N次
        
        # 新的协方差矩阵加上原本高斯点位置，也就是将旋转后的样本点添加到原始点的位置
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N,1) 
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1)/(0.8*N)) # 生成新的缩放参数 /1.6
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1) # 将旋转矩阵重复N次
        new_intensity = self._intensity[selected_pts_mask].repeat(N)

        # 调用另一个方法 densification_postfix，该方法对新生成的点执行后处理操作
        self.densification_postfix(new_xyz, new_intensity, new_scaling, new_rotation)

        # 创建一个修剪（pruning）的过滤器，将新生成的点添加到原始点的掩码之后。根据修剪过滤器，修剪模型中的一些参数
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, grad_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # 对于每个点计算梯度的L2范数，如果≥指定的阈值则标为true
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grad_abs, dim=-1) >= grad_abs_threshold, True, False)
        # 计算 mask 中为 True 的数量
        num_true = torch.sum(selected_pts_mask).item()
        print("Number of True values in selected_pts_mask:", num_true)
        print("Number of True values in selected_pts_mask_abs:", torch.sum(selected_pts_mask_abs).item())

        # 在上述掩码的基础上进一步过滤掉那些缩放大于一定百分比的场景范围的点。这样可以确保新添加的点不会太远离原始数据。
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
       
        # 根据掩码选取符合条件的点的其他特征。
        new_xyz = self._xyz[selected_pts_mask]
        new_intensity = self._intensity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_intensity, new_scaling, new_rotation)
    

    def add_densification_stat(self, viewspace_point_tensor, update_filter):
        # 打印张量的形状以进行调试
        # print("viewspace_point_tensor.shape:", viewspace_point_tensor.shape) #[30000,3]
        # print("update_filter.shape:", update_filter.shape)      # [30000]
        # print("self.xyz_gradient_accum shape: ", self.xyz_gradient_accum.shape) # [0]
        update_filter = update_filter.to(self.xyz_gradient_accum.device)
        tmp = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True) # torch.Size([30000, 1])
        # print('[debug] tmp shape:', tmp.shape)
        xyz =  self.xyz_gradient_accum[update_filter]
        
        # print('[debug] xyz shape:', xyz.shape)
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] += torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1


    from mayavi import mlab
    import numpy as np

    def visualize_gaussians(self, points, intensities, scales):
        # 创建一个新的图形窗口
        fig = mlab.figure(bgcolor=(0, 0, 0))
        print('points.shape: ', points.shape)
        print('intensities.shape: ', intensities.shape)
        print('scale.shape: ', scales.shape)


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
        mlab.axes()
        mlab.show()




def test_visualize():
    print('debug...')

    num_points = 200
    min_coord = [0, 0, 0]
    max_coord = [540, 540, 270]

    your_points = np.random.rand(num_points, 3) * (np.array(max_coord) - np.array(min_coord)) + np.array(min_coord)

    # your_points = np.random.rand(200, 3) * 200
    your_intensities = np.random.rand(num_points, 1)
    reconstruction = BasicPointCloud(points=your_points, intensities=your_intensities)

    model = GaussianModel_cryoET()
    model.create_from_fbp(reconstruction, intensity_threshold=0.1, init_size=3000, spatial_lr_scale=1.0, image_height=1024, image_width=1024, grid_size=8, volume_x=540, volume_y=540, volume_z=270)
    
    points = model._xyz.detach().cpu().numpy()
    intensities = model._intensity.detach().squeeze().cpu().numpy()
    scales = model._scaling.detach().cpu().numpy()

    print('visualize...')
    # model.visualize_gaussians(points, intensities, scales)


if __name__ == "__main__":
    test_visualize()