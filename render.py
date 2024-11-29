import torch
import torchvision
import os,sys
from tqdm import tqdm
from os import makedirs
sys.path.append("/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/cryoET-3dgs")

from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, DatasetParams, get_combined_args
from gaussian_renderer import render, GaussianModel_cryoET


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    # 构建渲染结果和GT保存路径
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, "ours_{}".format(iteration), "gt")

    # 确保渲染结果和GT保存路径存在
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 遍历所有视图进行渲染
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        # 调用render函数执行渲染，获取渲染结果
        rendering = render(view, gaussians, pipeline, background , kernel_size=0.1)["render"]

        # 获取视图的ground truth
        gt = view.original_image[0:3, :, :]

        # 保存渲染结果和ground truth为图像文件
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(model: ModelParams, iteration:int, pipeline:PipelineParams, data:DatasetParams, skip_train:bool, skip_test:bool):
    with torch.no_grad(): # 禁用梯度计算，因为在渲染过程中不需要梯度信息
        gaussians = GaussianModel_cryoET()
        scene = Scene(model, data, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if model.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 将背景颜色转化为张量，移到GPU上

        if not skip_train:
            render_set(model.model_path, "train", scene.loaded_iter, scene.getTrainTilts(), gaussians, pipeline, background) # 渲染训练数据集
            render_set(model.model_path, "test", scene.loaded_iter, scene.getTestTilts(), gaussians, pipeline, background) # 渲染测试数据集


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
    print("Rendering" + args.model_path)

    # Initialize system state
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), dp.extract(args),
                 args.skip_train, args.skip_test)