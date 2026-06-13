import torch
import sys
import numpy as np
from cryoET_3dgs.utils.loss_utils import ssim

# def mse(img1, img2):
#     return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def mse(img1, img2, mask=None):
    """MSE error

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    n_channel = img1.shape[1]
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        mask = mask.flatten(1).repeat(1, n_channel)
        mask = torch.where(mask != 0, True, False)

        mse = torch.stack(
            [
                (((img1[i, mask[i]] - img2[i, mask[i]])) ** 2).mean(0, keepdim=True)
                for i in range(img1.shape[0])
            ],
            dim=0,
        )

    else:
        mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return mse



@torch.no_grad()
def psnr(img1, img2, mask=None, pixel_max=1.0):
    """PSNR

    Args:
        img1 (_type_): [b, c, h, w]
        img2 (_type_): [b, c, h, w]
        mask (_type_, optional): [b, c, h, w]. Defaults to None.

    Returns:
        _type_: _description_
    """
    mse_out = mse(img1, img2, mask)
    psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
    if mask is not None:
        if torch.isinf(psnr_out).any():
            print(mse_out.mean(), psnr_out.mean())
            psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
            psnr_out = psnr_out[~torch.isinf(psnr_out)]

    return psnr_out


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def crop_to_common_volume(img1, img2):
    """从两个张量中截取中心区域，使它们大小一致（取最小公共大小）"""
    shape1 = np.array(img1.shape)
    shape2 = np.array(img2.shape)
    min_shape = np.minimum(shape1, shape2)

    def crop_center(img, target_shape):
        start = ((np.array(img.shape) - target_shape) // 2).astype(int)
        end = start + target_shape
        slices = tuple(slice(s, e) for s, e in zip(start, end))
        return img[slices]

    img1_cropped = crop_center(img1, min_shape)
    img2_cropped = crop_center(img2, min_shape)
    return img1_cropped, img2_cropped


@torch.no_grad()
def metric_vol(img1, img2, metric="psnr", pixel_max=1.0):
    """Metrics for volume. img1 must be GT."""
    assert metric in ["psnr", "ssim"]
    
    if isinstance(img2, np.ndarray):
        img1 = torch.from_numpy(img1.copy())
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.copy())

    if img1.shape != img2.shape:
        img1, img2 = crop_to_common_volume(img1, img2)
        
    if metric == "psnr":
        if pixel_max is None:
            pixel_max = img1.max()
        mse_out = torch.mean((img1 - img2) ** 2)
        psnr_out = 10 * torch.log10(pixel_max**2 / mse_out.float())
        return psnr_out.item(), None
    
    elif metric == "ssim":
        ssims = []
        for axis in [0, 1, 2]:
            results = []
            count = 0
            n_slice = img1.shape[axis]
            for i in range(n_slice):
                if axis == 0:
                    slice1 = img1[i, :, :]
                    slice2 = img2[i, :, :]
                elif axis == 1:
                    slice1 = img1[:, i, :]
                    slice2 = img2[:, i, :]
                elif axis == 2:
                    slice1 = img1[:, :, i]
                    slice2 = img2[:, :, i]
                else:
                    raise NotImplementedError
                if slice1.max() > 0:
                    result = ssim(slice1[None, None], slice2[None, None])
                    count += 1
                else:
                    result = 0
                results.append(result)
            results = torch.tensor(results)
            mean_results = torch.sum(results) / count
            ssims.append(mean_results.item())
        return float(np.mean(ssims)), ssims



@torch.no_grad()
def metric_proj(img1, img2, metric="psnr", axis=0, pixel_max=1.0):
    """Metrics for projection

    Args:
        img1 (_type_): [x, y, z]
        img2 (_type_): [x, y, z]
        pixel_max (float, optional): _description_. Defaults to 1.0.
    """
    assert axis in [0, 1, 2, None]
    assert metric in ["psnr", "ssim"]
    if isinstance(img2, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    n_slice = img1.shape[axis]

    results = []
    count = 0
    for i in range(n_slice):
        if axis == 0:
            slice1 = img1[i, :, :]
            slice2 = img2[i, :, :]
        elif axis == 1:
            slice1 = img1[:, i, :]
            slice2 = img2[:, i, :]
        elif axis == 2:
            slice1 = img1[:, :, i]
            slice2 = img2[:, :, i]
        else:
            raise NotImplementedError
        if slice1.max() > 0:
            slice1 = slice1 / slice1.max()
            slice2 = slice2 / slice2.max()
            if metric == "psnr":
                result = psnr(
                    slice1[None, None], slice2[None, None], pixel_max=pixel_max
                )
            elif metric == "ssim":
                result = ssim(slice1[None, None], slice2[None, None])
            else:
                raise NotImplementedError
            count += 1
        else:
            result = 0
        results.append(result)
    results = torch.tensor(results)
    mean_results = torch.sum(results) / count
    return mean_results.item(), results.tolist()
