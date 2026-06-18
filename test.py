import sys
import os
import torch
from PIL import Image
from tqdm import tqdm, trange
import torchvision
from time import time
# import cv2
import numpy as np
import concurrent.futures
import yaml
from argparse import ArgumentParser, Namespace
import mrcfile
from random import randint
import matplotlib.pyplot as plt
from cryoET_3dgs.arguments import ModelParams, PipelineParams, OptimizationParams, DatasetParams, ModelHiddenParams
from cryoET_3dgs.scene import Scene, GaussianModel_cryoET
from cryoET_3dgs.utils.image_utils import psnr, time2file_name, metric_vol, metric_proj

from cryoET_3dgs.utils.loss_utils import l1_loss, l1_mask_loss, l2_loss, lpips_loss, fourier_loss, fourier_loss_dynamic, contrast_loss, ssim
from cryoET_3dgs.utils.general_utils import safe_state
from cryoET_3dgs.utils.extra_utils import o3d_knn, weighted_l2_loss_v2, count_zero_voxels, voxel_density_statistics
from cryoET_3dgs.gaussian_renderer import render, query, query_chunk, network_gui

def t2a(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor


# def evaluate_volume(
#     save_path,
#     name,
#     gaussians: GaussianModel_cryoET,
#     pipeline: PipelineParams,
# ):
#     """Evaluate volume reconstruction."""
#     slice_save_path = os.path.join(save_path, name)
#     os.makedirs(slice_save_path, exist_ok=True)

#     query_pkg = query(
#         gaussians,
#         [0,0,0],  # center
#         [512,512,180], # nVoxel
#         [1,1,1], # sVoxel
#         pipeline,
#     )
#     vol_pred = query_pkg["vol"]
#     # print('[debug] vol_pred min & max = ', vol_pred.min(), vol_pred.max())
#     print('[debug] vol_pred.shape = ', vol_pred.shape)


#     multithread_write(
#         [vol_pred[..., i][None] for i in range(vol_pred.shape[2])],
#         slice_save_path,
#         "_pred",
#     )
#     np.save(os.path.join(save_path, "vol_pred.npy"), t2a(vol_pred))
   
#     sitk.WriteImage(
#         sitk.GetImageFromArray(t2a(vol_pred).transpose(2, 0, 1)),
#         os.path.join(save_path, "vol_pred.nii.gz"),
#     )



def evaluate_volume(
    save_path,
    name,
    gaussians: GaussianModel_cryoET,
    pipeline: PipelineParams,
    voxel_size,  
    volume_size,
    if_whitebg,
):
    """Evaluate volume reconstruction."""
    if gaussians.get_xyz.shape[0] > 100000:
        query_pkg = query_chunk(
            gaussians,
            [0,0,0],  # center
            volume_size, # nVoxel
            voxel_size, # sVoxel
            pipeline,
            # invert_intensity=True,
        )
    else:
        # 如果高斯点数量较少，直接使用query函数
        query_pkg = query(
            gaussians,
            [0,0,0],  # center
            volume_size, # nVoxel
            voxel_size, # sVoxel
            pipeline,
        )
    vol_pred = query_pkg["vol"]
    vol_post = query_pkg["vol_post"]
    count_zero_voxels(vol_pred)
    vol_pred_np = t2a(vol_pred).astype(np.float32)
    vol_post_np = t2a(vol_post).astype(np.float32)
    # print(f'[debug] vol_pred min:{vol_pred.min()} & max:{vol_pred.max()} & mean: {vol_pred.mean()}' )

    # 保存
    mrc_path = os.path.join(save_path, "vol_pred_inverted.mrc")
    post_mrc_path = os.path.join(save_path, "vol_postprocess.mrc")
    inverted_mrc_path = os.path.join(save_path, "vol_post_inverted.mrc")
    slice_save_path = os.path.join(save_path, name)
    post_slice_save_path = os.path.join(save_path, name, "postprocess")
    os.makedirs(slice_save_path, exist_ok=True)
    os.makedirs(post_slice_save_path, exist_ok=True)
    
    # ================= 存储MRC格式 =================       
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        max_val = vol_pred_np.max()
        min_val = vol_pred_np.min()
        inverted_vol = max_val - vol_pred_np + min_val
        # MRC格式需要轴顺序为 (Z, Y, X) , 保证内存连续性 
        mrc_data = np.ascontiguousarray(inverted_vol.transpose(2,0,1)) 
        mrc.set_data(mrc_data)
 
        # 设置头部信息
        mrc.header.map = mrcfile.constants.MAP_ID
        mrc.header.mode = 2  # 32-bit float
        # 设置物理信息
        mrc.header.cella = (
            mrc_data.shape[2] * voxel_size[0],
            mrc_data.shape[1] * voxel_size[1],
            mrc_data.shape[0] * voxel_size[2]
        )
        mrc.header.nx, mrc.header.ny, mrc.header.nz = mrc_data.shape[2], mrc_data.shape[1], mrc_data.shape[0]
        mrc.header.mx, mrc.header.my, mrc.header.mz = mrc.header.nx, mrc.header.ny, mrc.header.nz
        mrc.update_header_stats()
    

    with mrcfile.new(post_mrc_path, overwrite=True) as mrc:
        # MRC格式需要轴顺序为 (Z, Y, X) , 保证内存连续性 
        mrc_data = np.ascontiguousarray(vol_post_np.transpose(2,0,1)) 
        mrc.set_data(mrc_data)
 
        # 设置头部信息
        mrc.header.map = mrcfile.constants.MAP_ID
        mrc.header.mode = 2  # 32-bit float
        # 设置物理信息
        mrc.header.cella = (
            mrc_data.shape[2] * voxel_size[0],
            mrc_data.shape[1] * voxel_size[1],
            mrc_data.shape[0] * voxel_size[2]
        )
        mrc.header.nx, mrc.header.ny, mrc.header.nz = mrc_data.shape[2], mrc_data.shape[1], mrc_data.shape[0]
        mrc.header.mx, mrc.header.my, mrc.header.mz = mrc.header.nx, mrc.header.ny, mrc.header.nz
        mrc.update_header_stats()
    
    with mrcfile.new(inverted_mrc_path, overwrite=True) as mrc:
        # 如果数据未归一化，改用以下方式反转：
        max_val = vol_post_np.max()
        min_val = vol_post_np.min()
        inverted_vol = max_val - vol_post_np + min_val
        mrc_data = np.ascontiguousarray(inverted_vol.transpose(2,0,1)) 
        mrc.set_data(mrc_data)
        # 设置头部信息
        mrc.header.map = mrcfile.constants.MAP_ID
        mrc.header.mode = 2  # 32-bit float
        # 设置物理信息
        mrc.header.cella = (
            mrc_data.shape[2] * voxel_size[0],
            mrc_data.shape[1] * voxel_size[1],
            mrc_data.shape[0] * voxel_size[2]
        )
        mrc.header.nx, mrc.header.ny, mrc.header.nz = mrc_data.shape[2], mrc_data.shape[1], mrc_data.shape[0]
        mrc.header.mx, mrc.header.my, mrc.header.mz = mrc.header.nx, mrc.header.ny, mrc.header.nz
        mrc.update_header_stats()


    def auto_contrast(img_16bit):
        # 计算 1% 和 99% 分位数
        vmin = np.percentile(img_16bit, 0.1)
        vmax = np.percentile(img_16bit, 99.9)
        
        # 移除 NaN 或 Inf（如果有）
        img_16bit = np.nan_to_num(img_16bit, nan=0.0, posinf=vmax, neginf=vmin)
        
        # 处理 vmin == vmax 的情况（例如全黑或全白图像）
        if vmax == vmin:
            return np.zeros_like(img_16bit, dtype=np.uint8)  # 返回全黑图像
        
        # 归一化并缩放到 [0, 255]
        normalized = (img_16bit - vmin) / (vmax - vmin)
        return np.clip(normalized * 255, 0, 255).astype(np.uint8)
    
    
    def write_slice(image_data, index, save_dir):
        # 生成并保存slice
        image_data = auto_contrast(image_data)
        Image.fromarray(image_data).save(os.path.join(save_dir, f"{index:05d}_pred.png"))
       
    slice_data = [vol_pred_np[...,i] for i in range(vol_pred_np.shape[2])]
    slice_data_post = [vol_post_np[...,i] for i in range(vol_post_np.shape[2])]
    # 切片保存
    multithread_write(slice_data, slice_save_path, write_slice)
    multithread_write(slice_data_post, post_slice_save_path, write_slice)


def multithread_write(image_list, path, write_callback):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()*2)
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_callback, image, index, path))
        
    # 等待所有任务完成
    for task in concurrent.futures.as_completed(tasks):
        if task.exception():
            print(f"切片保存错误: {task.exception()}")
    executor.shutdown()


def testing(
    model: ModelParams,
    opt:OptimizationParams, 
    pipeline: PipelineParams,
    data:DatasetParams, 
    hyper:ModelHiddenParams,
    iteration: int,
    args
):
    gaussians = GaussianModel_cryoET(disable_xyz_log_activation=True)
    gaussians.load_ply("output/shrec_model0/2025_02_23_14_23_36/gaussians/iteration_4000/gaussian_points_rgb.ply")
    intensities = gaussians.get_intensity
    scales = gaussians.get_scaling
    # print(f'[debug] intensity max = {intensities.max()}, min={intensities.min()}')
    # print(f'[debug] scale max = {scales.max()}, min = {scales.min()}')

    xyz = gaussians.get_xyz
    # Calculate max and min values along each dimension
    max_values, _ = torch.max(xyz, dim=0)
    min_values, _ = torch.min(xyz, dim=0)
    max_x, max_y, max_z = max_values
    min_x, min_y, min_z = min_values
    # print(f"\nMax values:  x: {max_x.item()}, y: {max_y.item()}, z: {max_z.item()}")
    # print(f"\nMin values:  x: {min_x.item()}, y: {min_y.item()}, z: {min_z.item()}")

    save_path = os.path.join(
        model.model_path,
        "voxelization",
        "iter_{}".format(iteration),
    )

    evaluate_volume(
                save_path,
                "reconstruction",
                gaussians,
                pipeline,
                voxel_size=(1, 1, 1),  
                volume_size=[data.volume_x,data.volume_y,data.volume_z],
                if_whitebg=model.white_background, 
            )



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dp = DatasetParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--config', type=str, default='config/10164.yaml', help='Path to the configuration file')
    # parser.add_argument('--config', type=str, default='config/shrec_model0.yaml', help='Path to the configuration file')

    parser.add_argument("--iteration", default=5000, type=int)
    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)
    if args.model_path == "":
        args.model_path = "output/shrec_model0/"

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    print(args)

    with torch.no_grad():
        testing(
            mp.extract(args),op.extract(args),
            pp.extract(args), dp.extract(args), hp.extract(args),
            args.iteration, args
        )

    print("\nTesting complete.")

