import numpy as np
import matplotlib
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from scipy.spatial.transform import Rotation
from tqdm import trange
import open3d as o3d
import sys
from skimage import measure
import cv2

sys.path.append("./")
from cryoET_3dgs.utils.general_utils import t2a


def show_two_slice(
    slice1,
    slice2,
    title1,
    title2,
    cmap="viridis",
    vmax=None,
    vmin=None,
    vmax2=None,
    vmin2=None,
    gamma=1.0,
    save=False,
    no_diff=False,
):
    if save:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
    if torch.is_tensor(slice1):
        slice1 = t2a(slice1)
    if torch.is_tensor(slice2):
        slice2 = t2a(slice2)
    if not no_diff:
        diff = np.abs(slice1 - slice2) ** gamma
        fig, ax = plt.subplots(1, 3, figsize=(13, 5))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    img1 = ax[0].imshow(
        slice1,
        cmap=cmap,
        vmin=slice1.min() if vmin is None else vmin,
        vmax=slice1.max() if vmax is None else vmax,
    )
    ax[0].title.set_text(title1)
    cax0 = make_axes_locatable(ax[0]).append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(img1, cax=cax0)

    img2 = ax[1].imshow(
        slice2,
        cmap=cmap,
        vmin=slice2.min() if vmin2 is None else vmin2,
        vmax=slice2.max() if vmax2 is None else vmax2,
    )
    ax[1].title.set_text(title2)
    cax1 = make_axes_locatable(ax[1]).append_axes("right", size="5%", pad=0.1)
    cbar2 = fig.colorbar(img2, cax=cax1)

    if not no_diff:
        img3 = ax[2].imshow(diff, cmap=cmap, vmin=0.0)
        ax[2].title.set_text("error")
        cax2 = make_axes_locatable(ax[2]).append_axes("right", size="5%", pad=0.1)
        cbar3 = fig.colorbar(img3, cax=cax2)

    plt.tight_layout()

    if save:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = np.array(data.reshape(fig.canvas.get_width_height()[::-1] + (4,)))[
            ..., :3
        ]
        plt.close()
        return data
    else:
        plt.show()
        plt.close()
        return
    
    
def show_two_slice_with_blending(
    slice1,
    slice2,
    title1,
    title2,
    cmap="viridis",
    vmax=None,
    vmin=None,
    vmax2=None,
    vmin2=None,
    gamma=1.0,
    save=False,
    no_diff=False,
    weight1=0.5,  # Added weight parameter for slice1
    weight2=0.5,  # Added weight parameter for slice2
):
    if save:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
    if torch.is_tensor(slice1):
        slice1 = t2a(slice1)
    if torch.is_tensor(slice2):
        slice2 = t2a(slice2)
    
    # Calculate weighted sum
    weighted_sum = weight1 * slice1 + weight2 * slice2
    
    if not no_diff:
        diff = np.abs(slice1 - slice2) ** gamma
        fig, ax = plt.subplots(1, 4, figsize=(16, 5))  # Changed to 4 subplots
    else:
        fig, ax = plt.subplots(1, 3, figsize=(13, 5))  # Changed to 3 subplots

    # Plot slice1
    img1 = ax[0].imshow(
        slice1,
        cmap=cmap,
        vmin=slice1.min() if vmin is None else vmin,
        vmax=slice1.max() if vmax is None else vmax,
    )
    ax[0].title.set_text(title1)
    cax0 = make_axes_locatable(ax[0]).append_axes("right", size="5%", pad=0.1)
    cbar1 = fig.colorbar(img1, cax=cax0)

    # Plot slice2
    img2 = ax[1].imshow(
        slice2,
        cmap=cmap,
        vmin=slice2.min() if vmin2 is None else vmin2,
        vmax=slice2.max() if vmax2 is None else vmax2,
    )
    ax[1].title.set_text(title2)
    cax1 = make_axes_locatable(ax[1]).append_axes("right", size="5%", pad=0.1)
    cbar2 = fig.colorbar(img2, cax=cax1)

    if not no_diff:
        # Plot difference
        img3 = ax[2].imshow(diff, cmap=cmap, vmin=0.0)
        ax[2].title.set_text("error")
        cax2 = make_axes_locatable(ax[2]).append_axes("right", size="5%", pad=0.1)
        cbar3 = fig.colorbar(img3, cax=cax2)
        
        # Plot weighted sum
        img4 = ax[3].imshow(
            weighted_sum,
            cmap=cmap,
            vmin=weighted_sum.min(),
            vmax=weighted_sum.max(),
        )
        ax[3].title.set_text(f"weighted sum ({weight1:.1f}:{weight2:.1f})")
        cax3 = make_axes_locatable(ax[3]).append_axes("right", size="5%", pad=0.1)
        cbar4 = fig.colorbar(img4, cax=cax3)
    else:
        # Plot weighted sum when no_diff is True
        img3 = ax[2].imshow(
            weighted_sum,
            cmap=cmap,
            vmin=weighted_sum.min(),
            vmax=weighted_sum.max(),
        )
        ax[2].title.set_text(f"weighted sum ({weight1:.1f}:{weight2:.1f})")
        cax2 = make_axes_locatable(ax[2]).append_axes("right", size="5%", pad=0.1)
        cbar3 = fig.colorbar(img3, cax=cax2)

    plt.tight_layout()

    if save:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = np.array(data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3])
        plt.close()
        return data
    else:
        plt.show()
        plt.close()
        return
    
    

def visualize_volume_pair(vol1, vol2, slice_axis='x', slice_idx=None, title=''):
    # vol1/vol2: [D, H, W]
    assert vol1.shape == vol2.shape
    if slice_axis == 'z':
        idx = slice_idx or vol1.shape[0] // 2
        img1 = vol1[idx].detach().cpu().numpy()
        img2 = vol2[idx].detach().cpu().numpy()
    elif slice_axis == 'y':
        idx = slice_idx or vol1.shape[1] // 2
        img1 = vol1[:, idx, :].detach().cpu().numpy()
        img2 = vol2[:, idx, :].detach().cpu().numpy()
    elif slice_axis == 'x':
        idx = slice_idx or vol1.shape[2] // 2
        img1 = vol1[:, :, idx].detach().cpu().numpy()
        img2 = vol2[:, :, idx].detach().cpu().numpy()
    else:
        raise ValueError("slice_axis must be 'x', 'y', or 'z'")

    plt.figure(figsize=(10, 5))
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("Prediction")
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title("Ground Truth")
    plt.show()
    plt.savefig('visualize_volume_pair.png')


def log_volume_to_tensorboard(tb_writer, vol1, vol2, iteration, prefix='volume', slice_idx=None):
    """
    使用TensorBoard记录体积切片对比
    
    参数:
        tb_writer: TensorBoard的SummaryWriter对象
        vol1: 预测体积 [D, H, W]
        vol2: GT体积 [D, H, W]
        iteration: 当前迭代次数
        prefix: TensorBoard中的标签前缀
        slice_idx: 指定切片索引(可选)
    """
    assert vol1.shape == vol2.shape
    
    # 自动确定中间切片
    slice_z = slice_idx if slice_idx is not None else vol1.shape[0] // 2
    slice_y = slice_idx if slice_idx is not None else vol1.shape[1] // 2
    slice_x = slice_idx if slice_idx is not None else vol1.shape[2] // 2
    
    # 准备各轴向切片
    slices = {
        'z': {
            'pred': vol1[slice_z, :, :].unsqueeze(0),  # 添加channel维度
            'gt': vol2[slice_z, :, :].unsqueeze(0),
            'diff': (vol1[slice_z, :, :] - vol2[slice_z, :, :]).abs().unsqueeze(0)
        },
        'y': {
            'pred': vol1[:, slice_y, :].unsqueeze(0),
            'gt': vol2[:, slice_y, :].unsqueeze(0),
            'diff': (vol1[:, slice_y, :] - vol2[:, slice_y, :]).abs().unsqueeze(0)
        },
        'x': {
            'pred': vol1[:, :, slice_x].unsqueeze(0),
            'gt': vol2[:, :, slice_x].unsqueeze(0),
            'diff': (vol1[:, :, slice_x] - vol2[:, :, slice_x]).abs().unsqueeze(0)
        }
    }
    
    # 记录到TensorBoard
    for axis in ['x', 'y', 'z']:
        # 原始图像对比
        comparison = torch.cat([
            slices[axis]['pred'], 
            slices[axis]['gt'], 
            slices[axis]['diff']
        ], dim=-1)  # 水平拼接
        
        tb_writer.add_image(
            f'{prefix}/{axis}_axis', 
            comparison,
            global_step=iteration,
            dataformats='CHW'
        )
        
        # 单独记录各分量以便单独查看
        tb_writer.add_image(
            f'{prefix}_components/{axis}_pred', 
            slices[axis]['pred'],
            global_step=iteration,
            dataformats='CHW'
        )
        tb_writer.add_image(
            f'{prefix}_components/{axis}_gt', 
            slices[axis]['gt'],
            global_step=iteration,
            dataformats='CHW'
        )
        tb_writer.add_histogram(
            f'{prefix}_hist/{axis}_diff', 
            slices[axis]['diff'],
            global_step=iteration
        )