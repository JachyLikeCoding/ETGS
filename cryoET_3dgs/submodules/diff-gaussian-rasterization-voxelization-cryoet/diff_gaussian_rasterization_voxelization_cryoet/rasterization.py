import torch
import torch.nn as nn
from typing import NamedTuple
from . import _C


'''
申明一些gaussian_rasterization的接口类和函数 主要还是一个作为pytorch和CUDA之间的API接口作用
'''

def cpu_deep_copy_tuple(input_tuple):
    # 用于将输入元组中的pytorch张量深度复制到CPU上，并返回复制后的元组
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
        means3D,
        means2D,
        intensities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D, # 高斯分布的三维坐标
        means2D, # 高斯分布的二维坐标（屏幕空间坐标）
        intensities, # 密度值
        scales, # 缩放因子
        rotations, # 旋转
        cov3Ds_precomp, # 预计算的三维协方差矩阵
        raster_settings, # 高斯光栅化的设置
    )

# 自定义的pytorch autograd 函数, 用于高斯光栅化的前向传播和反向传播
class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    # 用于定义前向渲染的规则，接受一系列输入参数，并调用 C++/CUDA 实现的 _C.rasterize_gaussians 方法进行高斯光栅化。
    def forward(
        ctx, # 上下文对象，用于保存计算中间结果以供反向传播使用。
        means3D,
        means2D,
        intensities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, #[0., 0., 0.]
            means3D,        # (P,3) 每个gaussian的XYZ均值
            intensities,    # (P,1) 密度值
            scales,         # 每个 3D Gaussian 的XYZ尺度
            rotations,      # 每个 3D Gaussian 的旋转四元组
            raster_settings.scale_modifier, # 1.0
            cov3Ds_precomp, # 提前计算好的每个3D Gaussian的协方差矩阵
            raster_settings.viewmatrix, # (4,4) 相机外参矩阵，world to camera
            raster_settings.projmatrix, # (4,4) 相机内参矩阵，camera to image
            raster_settings.kernel_size,
            raster_settings.subpixel_offset,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.volume_x,
            raster_settings.volume_y,
            raster_settings.volume_z,
            raster_settings.prefiltered, # False
            raster_settings.debug        # False
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # copy them before they can be corrupted
            try:
                num_rendered, intensity, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args) # C++/CUDA 光栅化计算的输出结果
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, intensity, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args) 

        # keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, intensities, scales, rotations, cov3Ds_precomp, radii, geomBuffer, binningBuffer, imgBuffer)
        
        return intensity, radii
        

    @staticmethod
    def backward(ctx, grad_outputs, _): 
        # 方法用于定义反向传播梯度下降的规则，接受输入的梯度，并调用 C++/CUDA 实现的 _C.rasterize_gaussians_backward 方法计算相关张量的梯度。
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        means3D, intensities, scales, rotations, cov3Ds_precomp, radii, geoBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        # 将梯度和其他输入参数重构为 C++ 方法所期待的形式
        args = (raster_settings.bg,
                means3D,
                radii,
                intensities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.kernel_size,
                raster_settings.subpixel_offset,
                grad_outputs,
                geoBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        # 注意，该函数中包含了对调试模式的处理，即如果启用了调试模式，则在计算前向和反向传播时保存了参数的副本，并在出现异常时将其保存到文件中以供调试。
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grad_means2D, grad_intensities, grad_mu, grad_means3D, grad_cov3Ds_precomp, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_intensities, grad_mu, grad_means3D, grad_cov3Ds_precomp, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        # print('[debug]--grad_intensities.shape :', grad_intensities.shape)
        # print('[debug]--intensities.shape :', intensities.shape)
        grad_intensities = grad_intensities.squeeze()
        
        # 梯度
        grads = (
            grad_means3D,
            grad_means2D,
            grad_intensities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads
    
# 定义了一个命名元组（NamedTuple），同于存储高斯渲染器的设置参数。
    # 通过使用命名元组，可以更清晰地定义和组织这些设置参数，并且可以像访问类属性一样访问命名元组的各个字段，使用命名元组可以提高代码的可读性和可维护性，
    # 使得对参数的访问更直观和方便。
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    volume_x: int
    volume_y: int
    volume_z: int
    kernel_size: float
    subpixel_offset: torch.Tensor
    bg: torch.Tensor
    scale_modifier: float # 缩放修正因子
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    prefiltered: bool # 是否进行预过滤
    debug: bool # 是否调试模式


# 用于高斯光栅化的pytorch模块。通过集成nn.Module类，使得该类可以方便地和其他pytorch模块组合使用，并利用pytorch的自动求导功能进行梯度计算和优化。
class GaussianRasterizer(nn.Module): # 定义了一个继承自nn.Module的类，表示高斯光栅化器。
    # 初始化方法，接受一个raster_settings参数，该参数包含了光栅化的设置（例如图像大小、背景颜色和投影矩阵等）。
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    # 标记可见点的方法。接受3D点的位置作为输入，并使用C++/CUDA代码执行视野剔除，返回一个布尔张量表示每个点是否可见
    def markVisible(self, positions):
        # Mark visible points with a boolean
        with torch.no_grad(): # 这里不计算梯度，因为这步只是用于判断可见性
            raster_settings = self.raster_settings
            # 调用一个C++/CUDA实现的函数来快速计算可见性
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix
            )
        return visible

    def forward(self, means3D, means2D, intensities, scales=None, rotations=None, cov3D_precomp=None):
        '''
        前向传播方法,用于将3D高斯光栅化操作渲染成2D图像。
        输入包括3D坐标、2D坐标、密度、缩放、旋转或预计算的3D协方差等
        means3D: 3D gaussian分布的中心位置
        means2D: 屏幕空间中3D高斯分布的预期位置, 用于梯度回传
        scales: 高斯分布的尺度参数
        rotations: 高斯分布的旋转参数
        cov3D_precomp: 预先计算的3D协方差矩阵
        return: 光栅化后的二维图像。
        '''
        raster_settings = self.raster_settings

        # 检查缩放旋转对 或 预计算的3D协方差是否同时提供，要求只提供其中一种。
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        # 如果某个输入参数为None，则将其初始化为空张量
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        
        # 调用C++/CUDA光栅化例程rasterize_gaussians，传递相应的输入参数和光栅化设置
        return rasterize_gaussians(
            means3D,
            means2D,
            intensities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )