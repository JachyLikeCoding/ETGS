import os
import sys
import torch
import datetime
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from random import randint

sys.path.append("/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/cryoET-3dgs")

from arguments import ModelParams, PipelineParams, OptimizationParams, DatasetParams
from scene import Scene, GaussianModel_cryoET
from utils.image_utils import psnr, time2file_name
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from gaussian_renderer import render, network_gui
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from torch.utils.tensorboard import SummaryWriter
TENSORBOARD_FOUND = True
# try:
#     import tensorflow as tf
#     print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False



def visualize_images(image, gt_image, iteration, tilt_id):
    """
    可视化渲染的图像和 GT 图像
    """
    # 将 tensor 转换为 numpy 数组并移动到 CPU
    image_np = image.detach().cpu().numpy().transpose(1, 2, 0)  # 假设图像格式为 (C, H, W)
    gt_image_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)

    # 可视化图像
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_np)
    ax[0].set_title(f'Rendered Image (Iteration: {iteration}, Tilt ID: {tilt_id})')
    ax[0].axis('off')

    ax[1].imshow(gt_image_np)
    ax[1].set_title('GT Image')
    ax[1].axis('off')

    plt.show()



def training(model:ModelParams, 
             opt:OptimizationParams, 
             pipe:PipelineParams, 
             data:DatasetParams, 
             testing_iterations, 
             saving_iterations, 
             checkpoint_iterations, 
             checkpoint, 
             debug_from):
    first_iter = 0

    tb_writer = prepare_output_and_logger(model)
    # tb_writer.info("Training parameters: {}".format(vars(opt)))
        
    # 定义超参数空间
    hparams = {
        'position_lr_init': args.position_lr_init,
        'scaling_lr': args.scaling_lr,
        'rotation_lr': args.rotation_lr,
        'intensity_lr': args.intensity_lr,
        'lambda_dssim': args.lambda_dssim,
        'percent_dense': args.percent_dense,
        'intensity_threshold': args.intensity_threshold,
        'grid_size': args.grid_size
    }

    loss = {"total": 0.0}

    # 记录超参数
    tb_writer.add_hparams(hparams, {'train_loss': loss["total"]})

    # 初始化高斯模型，用于表示场景中的每个点的3D高斯分布
    gaussians = GaussianModel_cryoET()
    
    # 初始化场景对象，加载数据集和每张图片对应的tilt的参数
    scene = Scene(model, data, gaussians)

    # 为高斯模型设置优化器和学习率调度器 设置GaussianModel的训练参数
    gaussians.training_setup(opt)

    # 如果提供了checkpoint，则从checkpoint加载模型参数并恢复训练进度
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Resuming from checkpoint {checkpoint} at iteration {first_iter}")
    
    bg_color = [1,1,1] if model.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing= True)

    trainTilts = scene.getTrainTilts().copy()
    testTilts = scene.getTestTilts().copy()
  
    # highresolution index
    highresolution_index = []
    for index, tilt in enumerate(trainTilts):
        if tilt.image_width >= 800:
            highresolution_index.append(index)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training Progress")
    progress_bar.update(first_iter)
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Get one tilt for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainTilts().copy()
        viewpoint_tilt = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # 如果达到调试起始点，启用调试模式
        if (iteration-1) == debug_from:
            pipe.debug = True

        # 根据设置决定是否使用随机背景颜色
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        #TODO ignore border pixels
        if model.ray_jitter:
            subpixel_offset = torch.rand((int(data.image_height), int(data.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None

        # 根据3D gaussian渲染当前视角的图像
        render_pkg = render(viewpoint_tilt, gaussians, pipe, bg, 
                            kernel_size=model.kernel_size, 
                            scaling_modifier=1, 
                            subpixel_offset=subpixel_offset)
        image, viewspace_points_tensor, visibility_filter, radii = (
            render_pkg["render"], 
            render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], 
            render_pkg["radii"]
        )
        # print('[debug] image shape: ', image.shape)
        # print('[debug] image min & max: ', image.min(), image.max())
        
        # 在渲染得到的图像和GT图像之间计算loss
        gt_image = viewpoint_tilt.original_image.cuda()
        # print('[debug] gt_image min & max: ', gt_image.min(), gt_image.max())

        # 动态调整去噪参数
        sigma = max(0.1, 0.5 * (1 - iteration / opt.iterations))  # 随着训练轮数增加，逐步减小去噪强度

        # 对 GT 图像进行去噪
        # denoised_gt_image = TF.gaussian_blur(gt_image, kernel_size=3, sigma=sigma)

        loss = {"total": 0.0}
        # 可视化当前图像
        # visualize_images(image, gt_image, iteration, viewpoint_tilt.tilt_id)

        # 确保像素值范围在 0 到 1 之间
        # noisy_image = torch.clamp(image, 0, 1)

        # 计算渲染图像和 GT 图像之间的 L1 loss
        Ll1 = l1_loss(image, gt_image)
        loss["render"] = Ll1
        loss["total"] += loss["render"]
        
        SSIM = ssim(image, gt_image)
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - SSIM
            loss["dssim"] = loss_dssim
            loss["total"] = (1.0 - opt.lambda_dssim) * loss["render"] * 1000 + opt.lambda_dssim * loss["dssim"] * 1000
        loss_total = loss["total"]

        print(f"\nTraining iteration:{iteration}| Tilt ID:{viewpoint_tilt.tilt_id}")
        print(f"\n[debug] L1 loss = {Ll1:.4f}, L_SSIM = {loss_dssim:.4f}, loss = {loss_total:.4f}")

        # print("Radii:", radii[:10])
        loss["total"].backward()
  
        iter_end.record()

        with torch.no_grad():
            # 更新进度条和损失显示
            ema_loss_for_log = 0.4 * loss["total"].item() + 0.6 * ema_loss_for_log
            if iteration % 20 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(20)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # log and save
            training_report(tb_writer, iteration, Ll1, SSIM, loss["total"], l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background, model.kernel_size))
            
            if iteration in saving_iterations: 
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # 在指定迭代区间内，对3D高斯模型进行增密和修剪
            if iteration < opt.densify_until_iter:
                # 将每个像素位置上的最大半径记录在 max_radii2D 中。这是为了密集化时进行修剪操作时的参考。
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 将和密集化相关的统计信息添加到gaussians模型中，包括视图空间点和可见性过滤器
                gaussians.add_densification_stat(viewspace_points_tensor, visibility_filter)

                # 对3D gaussian进行克隆或切分，并将密度小于一定阈值的3D gaussians进行删除
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 0.02 * data.image_width if iteration > opt.intensity_reset_interval else None
                    gaussians.densify_and_prune(max_grad=opt.densify_grad_threshold, min_intensity=data.intensity_threshold, extent = data.extent, max_screen_size=size_threshold)
                
                # 对3D gaussian的密度进行重置
                if iteration % opt.intensity_reset_interval == 0 or (model.white_background and iteration == opt.densify_from_iter):
                    print('reset intensity here......')
                    gaussians.reset_intensity()
                
            # 执行优化器的一步，并准备下一次迭代
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")



def prepare_output_and_logger(args):    
    if not args.model_path:
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        args.model_path = os.path.join("./output/", args.scene, date_time)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, SSIM, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/SSIM', SSIM.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('train_loss', loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestTilts()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainTilts()[idx % len(scene.getTrainTilts())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # Render image and normalize in one step
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg["render"]
                    # Normalize image once and in place
                    # image_max = 1.2
                    # image = image / image_max
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # gt_image = TF.gaussian_blur(gt_image, kernel_size=3, sigma=1.0)  # 使用高斯模糊去噪

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/render_iter_{iteration}", image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()



if __name__ == "__main__":

    print('torch.version.cuda: ', torch.version.cuda)
    parser = ArgumentParser(description="Training script parameters")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dp = DatasetParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/10164.yaml', help='Path to the configuration file')
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 5_000, 1_0000, 1_5000, 2_0000, 2_5000, 3_0000, 3_5000, 4_0000, 4_5000, 5_0000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 5_0000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(args)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    #根据 detect_anomaly 参数设置是否检测自动求导操作中的异常
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for key, value in config.items():
        setattr(args, key, value)

    training(mp.extract(args), op.extract(args), pp.extract(args), dp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")