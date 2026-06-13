import torch
import math
import sys
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel_cryoET
from arguments import PipelineParams, DatasetParams
from diff_gaussian_rasterization_voxelization_cryoet import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)


# 渲染场景函数，主要通过将高斯分布的点投影到2D屏幕上来生成渲染图像。
def render(viewpoint_tilt, pc:GaussianModel_cryoET, pipe:PipelineParams, data:DatasetParams, bg_color:torch.Tensor, kernel_size:float, use_deform=False, scaling_modifier=1.0, subpixel_offset=None, \
     tilt_no=None, iter=None, num_down_emb_c=5, num_down_emb_f=5):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU
    scaling_modifier: 可选的缩放修正值,用于调整3D高斯的尺度
    """

    # 创建一个和输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中的点的位置，用于后续计算对于屏幕空间中坐标的梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    try:
        screenspace_points.retain_grad() # 尝试保留张量的梯度。这是为了确保可以在反向传播过程中计算对于屏幕空间坐标的梯度。
    except:
        pass

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_tilt.image_height), int(viewpoint_tilt.image_width), 2), dtype=torch.float32, device="cuda")
    
    # 设置光栅化的配置
    raster_settings = GaussianRasterizationSettings(
        image_height = int(viewpoint_tilt.image_height),
        image_width = int(viewpoint_tilt.image_width),
        volume_x = int(data.volume_x),
        volume_y = int(data.volume_y),
        volume_z = int(data.volume_z),
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_tilt.sample_view_transform,
        projmatrix=viewpoint_tilt.full_proj_transform,
        prefiltered=False,
        debug=pipe.debug
    )

    # print('\n[debug] raster_settings:--------------------------')
    # print('\n....viewmatrix: \n', viewpoint_tilt.sample_view_transform)
    # print('\n....projmatrix: \n', viewpoint_tilt.full_proj_transform)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz        # [30000,3]
    means2D = screenspace_points # [30000,3]
    intensity = pc.get_intensity # [30000]
    # print(f'[debug] intensity max = {intensity.max()}')
    tilt_emb = torch.tensor(viewpoint_tilt.uid).to(means3D.device).repeat(means3D.shape[0],1)

    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else: 
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if use_deform:
        means3D_final, scales_final, rotations_final, intensity_final, extras = pc._deformation(means3D, scales, rotations,
                intensity, tilt_emb, tilt_no, pc, None, iter=iter, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f)
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        intensity_final = pc.intensity_activation(intensity_final)
        print(f'[debug] before deform: scales[0]={scales[0]}, rotations[0]={rotations[0]}, intensity[0]={intensity[0]}')
        print(f'[debug] after deform: scales_final[0]={scales_final[0]}, rotations_final[0]={rotations_final[0]}, intensity_final[0]={intensity_final[0]}')
    else:
        means3D_final = means3D
        scales_final = scales
        rotations_final = rotations
        intensity_final = intensity


    # 调用光栅化器，将高斯分布投影到屏幕上，获得渲染图像和每个高斯分布在屏幕上的半径
    rendered_image, radii = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        intensities = intensity_final,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp
    )

    rendered_image = torch.clamp((rendered_image - rendered_image.min()) / (rendered_image.max() - rendered_image.min()), 0.0, 1.0)

    # 返回一个字典，包含渲染的图像 [1,H,W]、屏幕空间坐标[p,3]、可见性过滤器（根据半径判断是否可见）以及每个高斯分布在屏幕上的半径[p]
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii>0,
            "radii": radii}



def render_simple(viewpoint_tilt, pc:GaussianModel_cryoET, data:DatasetParams, bg_color:torch.Tensor, kernel_size:float, scaling_modifier=1.0, subpixel_offset=None, override_color=None, debug=False):
    """
    Render the scene.
    Background tensor must be on GPU.
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_tilt.image_height), int(viewpoint_tilt.image_width), 2), dtype=torch.float32, device="cuda")
    
    # print(f'[debug] volume x y z: {int(data.volume_x)},{int(data.volume_y)},{int(data.volume_z)}')
    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_tilt.image_height),
        image_width=int(viewpoint_tilt.image_width),
        volume_x = int(data.volume_x),
        volume_y = int(data.volume_y),
        volume_z = int(data.volume_z),
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_tilt.sample_view_transform,
        projmatrix=viewpoint_tilt.full_proj_transform,
        prefiltered=False,
        debug=debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    intensity = pc.get_intensity # [30000]
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    #rendered_image, radii
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        intensities = intensity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = torch.clamp((rendered_image - rendered_image.min()) / (rendered_image.max() - rendered_image.min()), 0.0, 1.0)
    rendered_image = 1 - rendered_image
    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    }


def generate_background(shape, intensity=0.02, smooth_sigma=3):
    """
    生成平滑背景密度张量
    - shape: tuple, e.g. (512, 512, 250)
    - intensity: 背景整体强度
    - smooth_sigma: 平滑程度，越大越平滑
    """
    noise = torch.rand(shape, device='cuda') * intensity
    # 使用3D Gaussian滤波进行平滑
    noise = F.avg_pool3d(noise.unsqueeze(0).unsqueeze(0), kernel_size=smooth_sigma, stride=1, padding=smooth_sigma//2)
    return noise.squeeze()


 # 构造 3x3x3 高斯核
def create_gaussian_kernel_3d(kernel_size=3, sigma=1.0, device='cpu'):
    coords = torch.arange(kernel_size, device=device) - kernel_size // 2
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)
    gaussian_kernel = torch.exp(-0.5 * (grid ** 2).sum(dim=-1) / sigma**2)
    gaussian_kernel /= gaussian_kernel.sum()
    return gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)



def query(
    pc: GaussianModel_cryoET,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    """
    Query a volume by voxelizing 3D Gaussians and applying post-processing.
    """
    # print('\n[debug] Query a volume with voxelization.')
    # --- step 1: 设置体素化参数 ---
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]), nVoxel_y=int(nVoxel[1]), nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]), sVoxel_y=float(sVoxel[1]), sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    # print('voxel_settings:', voxel_settings)
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    # --- Step 2: 获取高斯参数
    means3D = pc.get_xyz
    intensity = pc.get_intensity
    scales = None
    rotations = None
    cov3D_precomp = None
    
    # intensity = 1.0 - intensity 

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # print(f'[debug] pc.scales: {scales.shape}, max = {means3D[0,:].max}')
    # print(f'[debug] pc.intensity: {intensity.shape}, max = {intensity.max()}, min = {intensity.min()}')
    # print(f'[debug] pc.means3D: {means3D.shape}')

    # --- Step 3: 体素化 ---
    vol_pred, radii = voxelizer(
        means3D=means3D,
        intensities=intensity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # ---- Step 4: 后处理：平滑 + 闭运算填孔 ---
    volume = vol_pred.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, D, H, W]

    mask = (volume > 0).float()
    kernel = create_gaussian_kernel_3d(kernel_size=3, sigma=0.8, device=volume.device)
    smoothed = F.conv3d(volume, kernel, padding=1)
    norm = torch.clamp(F.conv3d(mask, kernel, padding=1), min=1e-6)
    volume_smoothed = smoothed / norm  # shape: [1,1,D,H,W]

    # 小孔洞填补（闭运算）：膨胀后腐蚀
    binary_mask = (volume_smoothed > 0.01).float()
    morph_kernel = torch.ones((1, 1, 3, 3, 3), device=volume.device)
    dilated = F.conv3d(binary_mask, morph_kernel, padding=1) > 0
    closed = F.conv3d(dilated.float(), morph_kernel, padding=1) >= 20
    volume_closed = volume_smoothed * closed.float()

    # 去掉 batch 和 channel 维度
    vol_final = volume_closed.squeeze(0).squeeze(0)


    # --- Step 5: 归一化
    vol_postprocess = torch.log1p(vol_final * 10)

    epsilon = 1e-8  # 防止除以零
    min_val, max_val = vol_final.min(), vol_final.max()
    
    if (max_val - min_val) > epsilon:
        vol_final = (vol_final - min_val) / (max_val - min_val + epsilon)
    

    # --- Step 6: 统计信息 ---
    # 检查是否有太多接近零的值
    # num_zero = (vol_final < 1e-2).sum()
    # print(f"Number of values close to zero: {num_zero}")
    # print("\n After normalization:")
    # print(f"Min: {vol_final.min():.4f}, Max: {vol_final.max():.4f}, Mean: {vol_final.mean():.4f}")
    # print(f"[debug] Volume shape: {vol_final.shape}, Min: {vol_final.min():.4f}, Max: {vol_final.max():.4f}, Mean: {vol_final.mean():.4f}")
    # print(f"[debug] Volume postprocess shape: {vol_postprocess.shape}, Min: {vol_postprocess.min():.4f}, Max: {vol_postprocess.max():.4f}, Mean: {vol_postprocess.mean():.4f}")
    return {
        "vol": volume,
        "vol_post": vol_postprocess,  # 处理后的体积数据
        "radii": radii,
    }
    


def query_chunk(
    pc: GaussianModel_cryoET,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
    chunk_size=500000,  # 每次处理的高斯数量
    invert_intensity=False,
    smooth_sigma=0.8,
):
    """
    Query a volume by voxelizing 3D Gaussians and applying post-processing.
    """
    # print('\n[debug] Query a volume with voxelization.')
    # --- step 1: 设置体素化参数 ---
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]), nVoxel_y=int(nVoxel[1]), nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]), sVoxel_y=float(sVoxel[1]), sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    # print('voxel_settings:', voxel_settings)
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    # --- Step 2: 获取高斯参数
    means3D = pc.get_xyz
    intensity = pc.get_intensity
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # --- Step 3: 分块体素化 ---
    vol_accum = None
    
    for i in range(0, means3D.shape[0], chunk_size):
        # 处理每个chunk
        chunk_means3D = means3D[i:i+chunk_size]
        chunk_intensity = intensity[i:i+chunk_size]
        chunk_scales = scales[i:i+chunk_size] if scales is not None else None
        chunk_rotations = rotations[i:i+chunk_size] if rotations is not None else None
        chunk_cov3D_precomp = cov3D_precomp[i:i+chunk_size] if cov3D_precomp is not None else None

        # 调用体素化器
        vol_pred, radii = voxelizer(
            means3D=chunk_means3D,
            intensities=chunk_intensity,
            scales=chunk_scales,
            rotations=chunk_rotations,
            cov3D_precomp=chunk_cov3D_precomp,
        )
        # 累加结果
        # vol_pred = vol_pred.cpu()
        if vol_accum is None:
            vol_accum = vol_pred
        else:
            vol_accum += vol_pred
                    
    # ---- Step 4: 后处理：平滑 + 闭运算填孔 ---
    volume = vol_accum.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, D, H, W]

    mask = (volume > 0).float()
    kernel = create_gaussian_kernel_3d(kernel_size=3, sigma=smooth_sigma, device=volume.device)
    smoothed = F.conv3d(volume, kernel, padding=1)
    norm = torch.clamp(F.conv3d(mask, kernel, padding=1), min=1e-6)
    volume_smoothed = smoothed / norm  # shape: [1,1,D,H,W]

    # 小孔洞填补（闭运算）：膨胀后腐蚀
    binary_mask = (volume_smoothed > 0.01).float()
    morph_kernel = torch.ones((1, 1, 3, 3, 3), device=volume.device)
    dilated = F.conv3d(binary_mask, morph_kernel, padding=1) > 0
    closed = F.conv3d(dilated.float(), morph_kernel, padding=1) >= 20
    volume_closed = volume_smoothed * closed.float()

    # 去掉 batch 和 channel 维度
    vol_final = volume_closed.squeeze(0).squeeze(0)

    # --- Step 5: 归一化和反转
    vol_postprocess = torch.log1p(vol_final * 5)

    epsilon = 1e-8  # 防止除以零
    min_val, max_val = vol_postprocess.min(), vol_postprocess.max()
    vol_postprocess = (vol_postprocess - min_val) / (max_val - min_val + epsilon)
    vol_postprocess = torch.clamp(vol_postprocess, 0.0, 1.0)
    
    if invert_intensity:
        vol_postprocess = 1.0 - vol_postprocess
    
    # --- Step 6: 统计信息 ---
    # 检查是否有太多接近零的值
    # num_zero = (vol_final < 1e-2).sum()
    # print(f"Number of values close to zero: {num_zero}")
    # print("\n After normalization:")
    # print(f"Min: {vol_final.min():.4f}, Max: {vol_final.max():.4f}, Mean: {vol_final.mean():.4f}")
    
    # print(f"[debug] Volume shape: {vol_final.shape}, Min: {vol_final.min():.4f}, Max: {vol_final.max():.4f}, Mean: {vol_final.mean():.4f}")
    # print(f"[debug] Volume postprocess shape: {vol_postprocess.shape}, Min: {vol_postprocess.min():.4f}, Max: {vol_postprocess.max():.4f}, Mean: {vol_postprocess.mean():.4f}")
    
    return {
        "vol": vol_final,
        "vol_post": vol_postprocess,  # 处理后的体积数据
        "radii": radii,
    }
    

def extract_volume_chunk(gt_volume: torch.Tensor, 
                         center: torch.Tensor, 
                         n_voxels: torch.Tensor, 
                         s_voxel: torch.Tensor) -> torch.Tensor:
    """
    从 gt_volume 中提取以 center 为物理坐标中心、尺寸为 n_voxels * s_voxel 的子体积块。
    所有输入均为 CUDA Tensor，坐标顺序为 XYZ。

    参数:
        gt_volume: Tensor, 形状为 [D, H, W] (ZYX)
        center: Tensor, 形状为 [3]，物理世界中的中心点 (单位与 s_voxel 一致)
        n_voxels: Tensor, 形状为 [3]，表示在 X,Y,Z 方向上的voxel数量
        s_voxel: Tensor or float, 每个voxel的尺寸（物理单位）

    返回:
        sub_volume: Tensor, 形状为 [D_sub, H_sub, W_sub]
    """

    # Step 1: 转换体积中心坐标（物理空间中）
    vol_shape_xyz = torch.tensor(gt_volume.shape, device=gt_volume.device).float()
    vol_center_xyz = vol_shape_xyz / 2.0
    center_voxel = center / s_voxel + vol_center_xyz  # XYZ -> index
    # print(f"[extract_volume_chunk] vol_center_xyz: {vol_center_xyz}")
    # Step 2: 计算范围
    half_vox = n_voxels.float() / 2.0
    start_idx = torch.floor(center_voxel - half_vox).long()
    end_idx = start_idx + n_voxels.long()

    # Clamp 防止越界
    start_idx = torch.clamp(start_idx, min=0)
    end_idx = torch.clamp(end_idx, max=vol_shape_xyz.long())

    # ZYX 切片
    z_start, y_start, x_start = start_idx[2].item(), start_idx[1].item(), start_idx[0].item()
    z_end, y_end, x_end = end_idx[2].item(), end_idx[1].item(), end_idx[0].item()

    # Debug 信息
    # print(f"[extract_volume_chunk] center_world: {center}")
    # print(f"[extract_volume_chunk] center_voxel: {center_voxel}")
    # print(f"[extract_volume_chunk] start_idx: {start_idx}")
    # print(f"[extract_volume_chunk] end_idx: {end_idx}")
    # print(f"[extract_volume_chunk] slice (x,y,z): [{x_start}:{x_end}], [{y_start}:{y_end}], [{z_start}:{z_end}]")

    # 提取体积块
    return gt_volume[x_start:x_end, y_start:y_end, z_start:z_end]