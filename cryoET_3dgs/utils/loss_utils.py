import torch
import torch.nn.functional as F
import torch.fft
from torch.autograd import Variable
from math import exp
import lpips
import torchvision.transforms as transforms

def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1, img2)
    return loss.mean()

    
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


# def l1_mask_loss(network_output, gt, visible_mask):
#     # Calculate l1 loss in available region.
#     l1_loss = torch.abs((network_output - gt)).mean()
#     l1_mask_loss = l1_loss * visible_mask
#     return torch.mean(l1_mask_loss)

def l1_mask_loss(network_output, gt, visible_mask):
    l1_loss = torch.abs(network_output - gt)
    l1_mask_loss = l1_loss * visible_mask  # 确保形状匹配
    return torch.mean(l1_mask_loss)



def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    # 生成一维高斯核，并归一化
    gauss = torch.Tensor([exp(-(x - window_size//2) ** 2 / float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    # 创建二维高斯核
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 将生成的二维高斯核扩展以匹配输入数据中的通道数
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    # 最终将拓展后的核转换为pytorch变量并返回
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average):
    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    # 计算方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1 * mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        # 在不同维度上的平均值
        return ssim_map.mean(1).mean(1).mean(1)
    

# def tv_loss(x, k):
#     """
#     Calculate total variation loss.
#     x (n1, n2, n3, 1): 3d density field
#     k: relative weight
#     """
#     n1,n2,n3 = x.shape
#     tv_1 = torch.sum(torch.abs(x[:,:,:-1] - x[:,:,1:]))
#     tv_2 = torch.sum(torch.abs(x[:,:-1,:] - x[:,1:,:]))
#     tv_3 = torch.sum(torch.abs(x[:-1,:,:] - x[1:,:,:]))
#     tv = (tv_1 + tv_2 + tv_3) / (n1*n2*n3)
#     return tv * k

def tv_3d_loss(vol, reduction="sum"):
    # 确保输入没有非法值（可选调试检查）
    assert not torch.isnan(vol).any(), "Input contains NaN!"
    assert not torch.isinf(vol).any(), "Input contains Inf!"

    dx = torch.abs(vol[:-1, :, :] - vol[1:, :, :])
    dy = torch.abs(vol[:, :-1, :] - vol[:, 1:, :])
    dz = torch.abs(vol[:, :, :-1] - vol[:, :, 1:])

    tv = torch.sum(dx) + torch.sum(dy) + torch.sum(dz)

    if reduction == "mean":
        # total_elements = (
        #     (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2]
        #     + vol.shape[0] * (vol.shape[1] - 1) * vol.shape[2]
        #     + vol.shape[0] * vol.shape[1] * (vol.shape[2] - 1)
        # )
        d, h, w = vol.shape
        total_elements = (d - 1) * h * w + d * (h - 1) * w + d * h * (w - 1)

        # 处理分母为零的情况
        if total_elements == 0:
            return torch.tensor(0.0, device=vol.device)
        else:
            tv = tv / total_elements
    return tv


def fourier_loss(prediction, gt, lambda_f=0.01):
    prediction_freq = torch.fft.fft2(prediction)
    gt_freq = torch.fft.fft2(gt)

    prediction_freq_mag = torch.abs(prediction_freq)
    gt_freq_mag = torch.abs(gt_freq)

    loss_freq = torch.mean((prediction_freq_mag - gt_freq_mag)**2) * lambda_f

    return loss_freq

def fourier_loss_dynamic(prediction, gt, iteration, total_iterations, max_lambda_f=0.0001, min_lambda_f=0.00001):
    # Compute FFT of the prediction and the ground truth
    prediction_freq = torch.fft.fft2(prediction)
    gt_freq = torch.fft.fft2(gt)

    # Compute the magnitude of the frequency components
    prediction_freq_mag = torch.abs(prediction_freq)
    gt_freq_mag = torch.abs(gt_freq)

    # Compute dynamic lambda_f based on the current iteration
    lambda_f = max_lambda_f - (max_lambda_f - min_lambda_f) * (iteration / total_iterations)
    
    # Compute the Fourier loss
    loss_freq = torch.mean((prediction_freq_mag - gt_freq_mag)**2) * lambda_f

    return loss_freq



def contrast_loss(prediction, gt, kernel_size=3, lambda_c=1):
    # 使用Sobel算子计算图像的梯度
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(prediction.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(prediction.device)

    # 对图像进行梯度计算
    gradient_x_pred = F.conv2d(prediction.unsqueeze(0), sobel_x, padding=1)
    gradient_y_pred = F.conv2d(prediction.unsqueeze(0), sobel_y, padding=1)
    
    gradient_x_gt = F.conv2d(gt.unsqueeze(0), sobel_x, padding=1)
    gradient_y_gt = F.conv2d(gt.unsqueeze(0), sobel_y, padding=1)

    # 计算梯度的L2范数（即局部对比度）
    grad_pred = torch.sqrt(gradient_x_pred**2 + gradient_y_pred**2 + 1e-6)  # 加上小常数以避免除零错误
    grad_gt = torch.sqrt(gradient_x_gt**2 + gradient_y_gt**2 + 1e-6)
    
    # 计算对比度损失
    contrast_loss_value = torch.mean((grad_pred - grad_gt)**2) * lambda_c

    return contrast_loss_value


def contrast_loss_batch(prediction, gt, kernel_size=3, lambda_c=1):
    # 确保输入是 4D: [batch, channels, height, width]
    assert prediction.ndim == 4, f"Expected 4D input, got {prediction.shape}"
    assert gt.ndim == 4, f"Expected 4D input, got {gt.shape}"

    # 创建 Sobel 过滤器
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(prediction.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(prediction.device)

    # 适配通道数（如果有多个通道，需要扩展 filter）
    sobel_x = sobel_x.repeat(prediction.shape[1], 1, 1, 1)  # [C, 1, 3, 3]
    sobel_y = sobel_y.repeat(prediction.shape[1], 1, 1, 1)  # [C, 1, 3, 3]

    # 计算 Sobel 梯度
    gradient_x_pred = F.conv2d(prediction, sobel_x, padding=1, groups=prediction.shape[1])  # 保持通道独立
    gradient_y_pred = F.conv2d(prediction, sobel_y, padding=1, groups=prediction.shape[1])

    gradient_x_gt = F.conv2d(gt, sobel_x, padding=1, groups=gt.shape[1])
    gradient_y_gt = F.conv2d(gt, sobel_y, padding=1, groups=gt.shape[1])

    # 计算梯度的 L2 范数（局部对比度）
    grad_pred = torch.sqrt(gradient_x_pred ** 2 + gradient_y_pred ** 2)
    grad_gt = torch.sqrt(gradient_x_gt ** 2 + gradient_y_gt ** 2)

    # 计算批量对比度损失
    contrast_loss_value = torch.mean((grad_pred - grad_gt) ** 2) * lambda_c

    return contrast_loss_value