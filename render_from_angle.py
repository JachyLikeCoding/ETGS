import torch
import torchvision
import os,sys
from tqdm import tqdm
import mrcfile
import numpy as np
from os import makedirs
import matplotlib.pyplot as plt
from cryoET_3dgs.scene import Scene
from cryoET_3dgs.utils.general_utils import safe_state
from argparse import ArgumentParser
from cryoET_3dgs.arguments import ModelParams, PipelineParams, DatasetParams, get_combined_args
from cryoET_3dgs.gaussian_renderer import render, GaussianModel_cryoET


def render_set(model_path, name, iteration, views, gaussians, pipeline, data, background, if_white=False, axis_id=0):
    # 构建渲染结果和GT保存路径
    render_path = os.path.join(model_path, name, "ours_render_{}".format(iteration), "renders")
    mrc_data = []

    # 确保渲染结果和GT保存路径存在
    makedirs(render_path, exist_ok=True)

    # 遍历所有视图进行渲染
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 调用render函数执行渲染，获取渲染结果
        rendering = render(view, gaussians, pipeline, data, background, kernel_size=0.1)["render"]
        # vmax = torch.quantile(rendering,0.99)
        # rendering_norm = rendering / (vmax)
        # rendering_norm = torch.clamp(rendering_norm, min=0.0, max=1.0)
        rendering_norm = torch.clamp(rendering, min=0.0, max=1.0)

        if not if_white:
            rendering_norm = 1 - rendering_norm
        
        # 保存渲染结果和ground truth为图像文件
        torchvision.utils.save_image(rendering_norm, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        img_gray = rendering_norm[0:1, :, :]  # 取渲染结果的灰度通道
        img_gray = img_gray.cpu().numpy()  # 将张量转换为NumPy数组
        img_gray = img_gray.squeeze()  # 去除多余的维度
        mrc_data.append(img_gray)  # 将灰度图像添加到列表中

    if len(mrc_data) > 0:
        mrc_stack = np.stack(mrc_data, axis=0)  # 将所有灰度图像堆叠成一个三维数组
        mrc_stack = np.transpose(mrc_stack, (0, 2, 1))  # 这里交换了第1和第2维度
        # 将所有灰度图像数据保存为MRC格式
        mrc_path = os.path.join(render_path, f"renders_axis{axis_id}.mrc")
        with mrcfile.new(mrc_path, overwrite=True) as mrc:
            mrc.set_data(mrc_stack) 
        print(f"Rendered images saved to {render_path}")


def render_sets(model: ModelParams, iteration:int, pipeline:PipelineParams, data:DatasetParams, skip_train=False, skip_test=False, if_white=False, axis_id=0):
    with torch.no_grad(): # 禁用梯度计算，因为在渲染过程中不需要梯度信息
        gaussians = GaussianModel_cryoET(disable_xyz_log_activation=True)
        scene = Scene(model, data, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if model.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 将背景颜色转化为张量，移到GPU上

        if not skip_train:
            render_set(model.model_path, "render_train", scene.loaded_iter, scene.getTrainTilts(), gaussians, pipeline, data, background, if_white, axis_id) # 渲染训练数据集
        if not skip_test:
            if not scene.test_tilts:
                print("No test tilts found, skipping test rendering.")
                return
            render_set(model.model_path, "render_test", scene.loaded_iter, scene.getTestTilts(), gaussians, pipeline, data, background, if_white, axis_id) # 渲染测试数据集


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    dp = DatasetParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    
    args.axis_id = 1
    
    # print("Rendering" + args.model_path)
    # Initialize system state
    safe_state(args.quiet)
    print(args)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), dp.extract(args), args.skip_train, args.skip_test, if_white=False, axis_id=args.axis_id)
    


    # python render.py --model_path /media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/cryoet-10643/2025_04_06_19_06_44/ --image_height 540 --image_width 540 --volume_x 540 --volume_y 540 --volume_z 540 --iteration 5000
    # /media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/cryoet-10643/2025_04_06_19_06_44


    # python render_from_angle.py --model_path /media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/shrec_model9/2025_05_17_21_24_04/ --image_height 2048 --image_width 2048 --volume_x 512 --volume_y 512 --volume_z 180

    # python render_from_angle.py --model_path /media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/cryoet-10453-TS082/2025_06_26_12_17_29/ --image_height 720 --image_width 511 --volume_x 720 --volume_y 511 --volume_z 256