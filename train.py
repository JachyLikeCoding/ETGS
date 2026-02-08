import os
import sys
import datetime
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from random import randint
from test import evaluate_volume
from display import visualize_images_with_mask
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
sys.path.append("/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/cryoET_3dgs")
from cryoET_3dgs.arguments import ModelParams, PipelineParams, OptimizationParams, DatasetParams, ModelHiddenParams
from cryoET_3dgs.scene import Scene, GaussianModel_cryoET
from cryoET_3dgs.utils.image_utils import psnr, time2file_name, metric_proj, metric_vol
from cryoET_3dgs.utils.loss_utils import l1_loss, tv_3d_loss, fourier_loss_dynamic, contrast_loss, ssim
from cryoET_3dgs.utils.general_utils import safe_state
from cryoET_3dgs.utils.extra_utils import o3d_knn, get_cuda_memory_usage
from cryoET_3dgs.gaussian_renderer import render, query, query_chunk, extract_volume_chunk, network_gui
from cryoET_3dgs.utils.plot_utils import show_two_slice, show_two_slice_with_blending, visualize_volume_pair, log_volume_to_tensorboard
from render_from_angle import render_set
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from torch.utils.tensorboard import SummaryWriter
TENSORBOARD_FOUND = True


def training(model:ModelParams, 
             opt:OptimizationParams, 
             pipe:PipelineParams, 
             data:DatasetParams, 
             hyper:ModelHiddenParams,
             testing_iterations, 
             saving_iterations, 
             checkpoint_iterations, 
             checkpoint, 
             debug_from,
             args):
    tb_writer = prepare_output_and_logger(args, model, opt, pipe, data, hyper)
        
    hparams = {
        'position_lr_init': args.position_lr_init,
        'scaling_lr': args.scaling_lr,
        'scaling_lr_init': args.scaling_lr_init,
        'rotation_lr': args.rotation_lr,
        'rotation_lr_init': args.rotation_lr_init,
        'intensity_lr_init': args.intensity_lr_init,
        'intensity_lr': args.intensity_lr,
        'embedding_lr': args.embedding_lr,
        'lambda_dssim': args.lambda_dssim,
        'percent_dense': args.percent_dense,
        'intensity_threshold': args.intensity_threshold,
        'grid_size': args.grid_size
    }

    loss = {"total": 0.0}
    tb_writer.add_hparams(hparams, {'train_loss': loss["total"]})

    queryfunc = lambda x: query_chunk(
        x,
        [0, 0, 0],
        [data.volume_x,data.volume_y,data.volume_z],
        [1, 1, 1],
        pipe,
        invert_intensity=True,
    )

    scale_bound = None
    if model.scale_min > 0 and model.scale_max > 0:
        scale_bound = np.array([model.scale_min, model.scale_max]) * data.volume_x
        print(f'[debug] scale bound: {scale_bound}')

    gaussians = GaussianModel_cryoET(disable_xyz_log_activation=False, args=args, use_deformation=hyper.use_deform, scale_bound=scale_bound)
    scene = Scene(model, data, gaussians)
    bbox = scene.bbox
    gaussians.training_setup(opt)
    first_iter = 0
    
    if checkpoint:
        try:
            (model_params, first_iter) = torch.load(checkpoint, map_location="cuda")
            gaussians.restore(model_params, opt)
            print(f"Resuming from checkpoint {checkpoint} at iteration {first_iter}")
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint}: {e}")
            first_iter = 0

    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size], device="cuda")
        tv_vol_sVoxel = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    
    bg_color = [1,1,1] if model.white_background else [0,0,0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing= True)
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training Progress")
    progress_bar.update(first_iter)
    first_iter += 1
    print(f"...... Training begin!!! .....")

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        parma_groups = gaussians.optimizer.param_groups
        for i, param_group in enumerate(gaussians.optimizer.param_groups):
            current_lr = param_group["lr"]
            param_name = param_group.get("name", f"group_{i}")
            tb_writer.add_scalar(f'learning_rate/{param_name}', current_lr, iteration)
        
        # Choose viewpoint tilt and calculate loss
        viewpoint_stack = scene.getTrainTilts().copy()
        num_viewpoints = opt.num_viewpoints
        # print(f'\n....select {num_viewpoints} tilts for training....')

        if opt.num_viewpoints == 0 or opt.num_viewpoints > len(viewpoint_stack):
            num_viewpoints = len(viewpoint_stack)
        selected_viewpoints = random.sample(viewpoint_stack, min(num_viewpoints, len(viewpoint_stack)))

        if (iteration-1) == debug_from:
            pipe.debug = True
        
        if model.ray_jitter:
            subpixel_offset = torch.rand((int(data.image_height), int(data.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
        else:
            subpixel_offset = None

        # weights for each tilt
        weights = [viewpoint_tilt.weight for viewpoint_tilt in selected_viewpoints]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        tilt_no_list = []
        loss_total = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        total_Ll1 = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        total_dssim = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        total_freq = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        total_contrast = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        total_deform = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        total_smooth = torch.tensor(0.0, device="cuda", dtype=torch.float32)
        SSIM_total = torch.tensor(0.0, device="cuda", dtype=torch.float32)

        # Calculate loss for each selected tilt
        for i, viewpoint_tilt in enumerate(selected_viewpoints):
            tilt_no = viewpoint_tilt.uid
            tilt_no_list.append(tilt_no)

            with torch.cuda.amp.autocast():
                render_pkg = render(viewpoint_tilt, gaussians, pipe, data, bg, 
                                    kernel_size=model.kernel_size, 
                                    scaling_modifier=1, 
                                    subpixel_offset=subpixel_offset,
                                    tilt_no=tilt_no, 
                                    iter=iteration,
                                    use_deform=hyper.use_deform,
                                    num_down_emb_c=hyper.min_embeddings, num_down_emb_f=hyper.min_embeddings
                                    )
            image, viewspace_points_tensor, visibility_filter, radii = (
                render_pkg["render"], 
                render_pkg["viewspace_points"], 
                render_pkg["visibility_filter"], 
                render_pkg["radii"]
            )
            # upper = torch.quantile(image.flatten(), 0.995)
            # print(f'     upper = {upper}')
            # image = image.clamp(min=0.0, max=float(upper))
            
            if not model.white_background:
                image = 1 - image

            gt_image = viewpoint_tilt.original_image.cuda() # [1, h, w]
            gt_image = torch.mean(gt_image, dim=0, keepdim=True)  # 在通道维度取均值

            weight = normalized_weights[i]
            Ll1 = l1_loss(image, gt_image) * weight * 1000
            total_Ll1 += Ll1

            mask = viewpoint_tilt.mask
            mask_tensor = torch.from_numpy(mask).to("cuda").float().transpose(0, 1).unsqueeze(0)

            # 可视化当前图像
            if iteration % 1000 == 0:
                visualize_images_with_mask(image, gt_image, mask_tensor, iteration, viewpoint_tilt.tilt_id, viewpoint_tilt.image_name, args.model_path)

            assert image.shape == gt_image.shape, f"Prediction shape {image.shape} does not match ground truth {gt_image.shape}"
            SSIM = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            SSIM_total += SSIM
            if opt.lambda_dssim > 0:
                loss_dssim = (1.0 - SSIM) * opt.lambda_dssim * weight * 1000
                total_dssim += loss_dssim
            else:
                loss_dssim = 0.0

            if torch.isnan(image).any() or torch.isinf(image).any():
                print("Warning: prediction contains NaN or inf at iteration", iteration)

            Freq_loss = fourier_loss_dynamic(image, gt_image, iteration, total_iterations=opt.iterations) * weight * 100
            total_freq += Freq_loss
            Contrast_loss = contrast_loss(image, gt_image, 3, lambda_c=0.5) * weight * 100
            total_contrast += Contrast_loss

        if use_tv:
            # Randomly get tiny volume center
            block_size = tv_vol_nVoxel.float() * tv_vol_sVoxel
           
            min_center = bbox[0] + block_size / 2
            max_center = bbox[1] - block_size / 2
            tv_vol_center = min_center + (max_center - min_center) * torch.rand(3, device="cuda")

            vol_pred = query_chunk(
                gaussians, 
                tv_vol_center, 
                tv_vol_nVoxel, 
                tv_vol_sVoxel, 
                pipe,
                invert_intensity=True)["vol_post"]
            
            # GT volume chunk
            vol_gt = scene.vol_gt
            vol_gt_chunk = extract_volume_chunk(vol_gt, tv_vol_center, tv_vol_nVoxel, tv_vol_sVoxel)

            min_val, max_val = vol_gt_chunk.min(), vol_gt_chunk.max()
            vol_gt_normalized = (vol_gt_chunk - min_val) / (max_val - min_val + 1e-8)
            vol_gt_normalized = torch.clamp(vol_gt_normalized, 0.0, 1.0)
    
            if iteration % 500 == 0 and tb_writer:
                # visualize_volume_pair(vol_pred, vol_gt_normalized, slice_axis='x', slice_idx=vol_gt_normalized.shape[2] // 2, title=f"TV Loss Volume at Iteration {iteration}")
                log_volume_to_tensorboard(tb_writer, vol_pred, vol_gt_normalized, iteration, prefix='volume', slice_idx=None)
            
            # volume L1 loss
            volume_l1_loss = F.l1_loss(vol_pred, vol_gt_normalized) * 1000
            loss_tv = tv_3d_loss(vol_pred, reduction = "mean") * 1000

        else:
            volume_l1_loss = 0.0
            loss_tv = 0.0

        loss_total = (
            total_Ll1
            + opt.lambda_dssim * total_dssim
            + opt.lambda_tv * loss_tv + volume_l1_loss
            + total_freq + total_contrast
        )
        
        loss_total.backward()
        iter_end.record()

        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stat(viewspace_points_tensor, visibility_filter)

                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if gaussians.get_xyz.shape[0] > int(opt.max_gaussians_num):
                        print(f'[debug] Totally {gaussians.get_xyz.shape[0]} gaussians in the scene > the max num {opt.max_gaussians_num}, no densify here...')
                        gaussians.only_prune(data.intensity_threshold, extent=data.extent, bbox=bbox, max_num=opt.max_gaussians_num)
                    else:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, data.intensity_threshold, data.extent, 
                                                radii, opt.max_gaussians_num, iteration, bbox, args.model_path, tb_writer)
                    
            # Optimizer step            
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            
            # Voxelization
            if iteration in saving_iterations: 
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, data)
                save_path = os.path.join(model.model_path,"test","iter_{}".format(iteration))
                vol_size = [data.volume_x, data.volume_y, data.volume_z]
                evaluate_volume(
                    save_path,
                    "reconstruction",
                    gaussians,
                    pipe,
                    voxel_size=[1,1,1],
                    volume_size=vol_size,
                    if_whitebg=model.white_background
                )
                
                render_tilts = sorted(viewpoint_stack, key=lambda x: x.tilt_id)
                render_set(model.model_path, "train", iteration, render_tilts, gaussians, pipe, data, bg, if_white=model.white_background)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss_total.item():.{4}f}",
                        "pts": f"{gaussians.get_intensity.shape[0]}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # log and save
            if iteration % 50 == 0 or iteration <= 10:
                training_report(tb_writer, iteration, total_Ll1.item(), total_dssim.item(), SSIM_total/num_viewpoints, 
                                total_freq.item(), total_contrast.item(), total_deform.item(), total_smooth.item(), loss_tv, volume_l1_loss, coverage_penalty, density_loss,
                                loss_total.item(), l1_loss, 
                                iter_start.elapsed_time(iter_end), 
                                testing_iterations, scene, model,
                                queryfunc, render, 
                                (pipe, data, bg, model.kernel_size, hyper.use_deform))
                

def prepare_output_and_logger(args, model, opt, pipe, data, hyper):    
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        tb_writer.add_text("ModelParams", str(vars(model)))
        tb_writer.add_text("OptimizationParams", str(vars(opt)))
        tb_writer.add_text("PipelineParams", str(vars(pipe)))
        tb_writer.add_text("DatasetParams", str(vars(data)))
        tb_writer.add_text("ModelHiddenParams", str(vars(hyper)))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, LSSIM, SSIM, Lfourier, Contrast_loss, Deform_loss, Smooth_loss, TV_loss, volume_l1_loss, Coverage_penalty, density_loss, loss, l1_loss, elapsed,
                    testing_iterations, scene:Scene, model, queryfunc, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/SSIM_loss', LSSIM, iteration)
        tb_writer.add_scalar('train_loss_patches/Contrast_loss', Contrast_loss, iteration)
        tb_writer.add_scalar('train_loss_patches/Fourier_loss', Lfourier, iteration)
        tb_writer.add_scalar('train_loss_patches/Deform_loss', Deform_loss, iteration)
        tb_writer.add_scalar('train_loss_patches/Smooth_loss', Smooth_loss, iteration)
        tb_writer.add_scalar('train_loss_patches/TV_loss', TV_loss, iteration)
        tb_writer.add_scalar('train_loss_patches/volume_l1_loss', volume_l1_loss, iteration)
        tb_writer.add_scalar('train_loss_patches/Coverage_penalty', Coverage_penalty, iteration)
        tb_writer.add_scalar('train_loss_patches/density_loss', density_loss, iteration)
        tb_writer.add_scalar('train_loss_patches/SSIM', SSIM, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('train_loss', loss, iteration)
        mem_allocated, mem_reserved = get_cuda_memory_usage()
        tb_writer.add_scalar("Memory/CUDA Allocated (MB)", mem_allocated, iteration)
        tb_writer.add_scalar("Memory/CUDA Reserved (MB)", mem_reserved, iteration)


    # Report test and samples of training set
    if iteration in testing_iterations:
        eval_save_path = os.path.join(args.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
            
        # torch.cuda.empty_cache()
        if tb_writer:
            tb_writer.add_histogram("scene/density_histogram", scene.gaussians.get_intensity, iteration)
            tb_writer.add_histogram(
                    "scene/scale_histogram", scene.gaussians.get_scaling, iteration
                )
        validation_configs = ({'name': 'test', 'tilts' : scene.getTestTilts()}, 
                              {'name': 'train', 'tilts' : [scene.getTrainTilts()[idx % len(scene.getTrainTilts())] for idx in range(5, 30, 5)]})
        
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config['tilts'] and len(config['tilts']) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                image_show_2d = []

                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['tilts']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, tilt_no=viewpoint.uid, iter=iteration)
                    image = render_pkg["render"]
                    if not model.white_background:
                        image = 1 - image

                    gt_image = viewpoint.original_image.cuda()

                    images.append(image)
                    gt_images.append(gt_image)

                    if tb_writer:
                        tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/render_iter_{iteration}", image.repeat(3,1,1).unsqueeze(0), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/ground_truth", gt_image.repeat(3,1,1).unsqueeze(0), global_step=iteration)
                        
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    vmin2=image[0].min() if iteration != 1 else None,
                                    vmax2=image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr", axis=0)
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim", axis=0)
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    os.path.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                psnr_test /= len(config['tilts'])
                l1_test /= len(config['tilts'])          
                print("\n[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f}".format(iteration, config['name'], l1_test, psnr_test))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        vol_pred = queryfunc(scene.gaussians)["vol_post"]
        vol_pred = (vol_pred - vol_pred.min())/(vol_pred.max() - vol_pred.min())
        # print(f'\n[debug] vol pred min: {vol_pred.min()}, vol max: {vol_pred.max()}, vol mean: {vol_pred.mean()}, vol median: {vol_pred.median()}')

        vol_gt = scene.vol_gt
        vol_gt = vol_gt.to(vol_pred.device) 
        
        gt_mean, gt_std = vol_gt.mean(), vol_gt.std()
        pred_mean, pred_std = vol_pred.mean(), vol_pred.std()

        vol_pred = (vol_pred - pred_mean) * (gt_std/(pred_std+1e-8)) + gt_mean
        vol_pred = torch.clamp(vol_pred, 0, 1)  # 保持合理范围

        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }
        with open(os.path.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_two_slice_with_blending(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"3D slice {i} gt",
                        f"3D slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        vmin2=vol_pred[..., i].min(),
                        vmax2=vol_pred[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )
        # torch.cuda.empty_cache()



if __name__ == "__main__":
    print('torch.version.cuda: ', torch.version.cuda)
    parser = ArgumentParser(description="Training script parameters")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dp = DatasetParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=True)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[ 1000, 2000, 3000, 4000, 5000, 6000, 8000, 1_0000, 12000, 1_5000, 2_0000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 1000, 2000, 4000, 6000, 8000, 1_0000, 1_5000, 2_0000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    # parser.add_argument('--config', type=str, default='config/shrec_model9.yaml', help='Path to the configuration file')
    # parser.add_argument('--config', type=str, default='config/shrec_model0.yaml', help='Path to the configuration file')
    # parser.add_argument('--config', type=str, default='config/10643.yaml', help='Path to the configuration file')
    # parser.add_argument('--config', type=str, default='config/10453.yaml', help='Path to the configuration file')
    # parser.add_argument('--config', type=str, default='config/11058.yaml', help='Path to the configuration file')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    
    if args.model_path == "":
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        args.model_path = os.path.join("/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/output/", args.scene, date_time)
        
    print(args)
    print("Optimizing " + args.model_path)

    training(mp.extract(args), op.extract(args), pp.extract(args), dp.extract(args), hp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    print("\nTraining complete.")