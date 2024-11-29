import torch
import math
from scene.gaussian_model import GaussianModel_cryoET
from arguments import PipelineParams
from diff_gaussian_rasterization_cryoet import GaussianRasterizationSettings, GaussianRasterizer


# 这段代码是一个用于渲染场景的函数，主要通过将高斯分布的点投影到2D屏幕上来生成渲染图像。
def render(viewpoint_tilt, pc:GaussianModel_cryoET, pipe, bg_color:torch.Tensor,  kernel_size:float, scaling_modifier=1.0, subpixel_offset=None ):
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
    # print('\n....tilt uid: ', viewpoint_tilt.uid)
    # print('\n........viewmatrix: \n', viewpoint_tilt.sample_view_transform)
    # print('\n........projmatrix: \n', viewpoint_tilt.full_proj_transform)


    # 创建一个高斯光栅化器对象，用于将高斯分布投影到屏幕上
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取高斯分布的三维坐标、屏幕空间坐标和密度
    means3D = pc.get_xyz        # [30000,3]
    means2D = screenspace_points # [30000,3]
    intensity = pc.get_intensity # [30000]

    # 如果提供了预先计算的3D协方差矩阵，则使用它。否则将由光栅化器根据尺度和旋转进行计算。
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else: 
        # 获取缩放和旋转信息，对应的就是3D高斯的协方差矩阵了
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 调用光栅化器，将高斯分布投影到屏幕上，获得渲染图像和每个高斯分布在屏幕上的半径
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        intensities = intensity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )
    
    # 返回一个字典，包含渲染的图像 [1,H,W]、屏幕空间坐标[p,3]、可见性过滤器（根据半径判断是否可见）以及每个高斯分布在屏幕上的半径[p]
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii>0,
            "radii": radii}





def render_simple(viewpoint_tilt, pc:GaussianModel_cryoET, bg_color:torch.Tensor, kernel_size:float, scaling_modifier=1.0, subpixel_offset=None, override_color=None, debug=False):
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
  
    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_tilt.image_height),
        image_width=int(viewpoint_tilt.image_width),
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

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
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

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    }