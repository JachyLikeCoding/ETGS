#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    intensities : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1) # 将每一个三维点拓展成齐次坐标
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getSample2View(R, T, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = T * scale + translate
    return np.float32(Rt)


def getOrthographicProjectionMatrix(znear, zfar):
    """
    Generate an orthographic projection matrix.
    """
    ortho_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],  # 保持x坐标不变
        [0.0, 1.0, 0.0, 0.0],  # 保持y坐标不变
        [0.0, 0.0, 2.0/(zfar-znear), -(znear+zfar)/(zfar-znear)],  # z坐标映射到[-1, 1]区间
        [0.0, 0.0, 0.0, 1.0]   # 齐次坐标分量
    ])
    # ortho_matrix = np.array([
    #     [1.0, 0.0, 0.0, 0.0],  # 保持x坐标不变
    #     [0.0, 1.0, 0.0, 0.0],  # 保持y坐标不变
    #     [0.0, 0.0, 1.0, 0.0],  # z坐标映射到[-1, 1]区间
    #     [0.0, 0.0, 0.0, 1.0]   # 齐次坐标分量
    # ])

    return torch.tensor(ortho_matrix, dtype=torch.float32)



def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str = "XYZ"):
    """
    将欧拉角转换为旋转矩阵。
    参数:
        euler_angles (torch.Tensor): 欧拉角，形状为 [3] 或 [batch_size, 3]，单位为弧度。
        convention (str): 欧拉角的旋转顺序，例如 "XYZ" 表示先绕X轴旋转 再绕Y轴 最后绕Z轴。
    返回:
        torch.Tensor: 旋转矩阵，形状为 [3, 3] 或 [batch_size, 3, 3]。
    """
    if euler_angles.dim() == 1:
        euler_angles = euler_angles.unsqueeze(0)  # 转换为 [1, 3]
    batch_size = euler_angles.shape[0]

    # 提取欧拉角
    x = euler_angles[:, 0]  # 绕X轴旋转
    y = euler_angles[:, 1]  # 绕Y轴旋转
    z = euler_angles[:, 2]  # 绕Z轴旋转

    # 计算旋转矩阵的每个分量
    cos_x, sin_x = torch.cos(x), torch.sin(x)
    cos_y, sin_y = torch.cos(y), torch.sin(y)
    cos_z, sin_z = torch.cos(z), torch.sin(z)

    # 创建单位矩阵
    Rx = torch.eye(3, device=euler_angles.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Ry = torch.eye(3, device=euler_angles.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Rz = torch.eye(3, device=euler_angles.device).unsqueeze(0).repeat(batch_size, 1, 1)

    # 绕X轴旋转
    Rx[:, 1, 1] = cos_x
    Rx[:, 1, 2] = -sin_x
    Rx[:, 2, 1] = sin_x
    Rx[:, 2, 2] = cos_x

    # 绕Y轴旋转
    Ry[:, 0, 0] = cos_y
    Ry[:, 0, 2] = sin_y
    Ry[:, 2, 0] = -sin_y
    Ry[:, 2, 2] = cos_y

    # 绕Z轴旋转
    Rz[:, 0, 0] = cos_z
    Rz[:, 0, 1] = -sin_z
    Rz[:, 1, 0] = sin_z
    Rz[:, 1, 1] = cos_z

    # 根据旋转顺序组合旋转矩阵
    if convention == "XYZ":
        R = torch.bmm(Rz, torch.bmm(Ry, Rx))
    elif convention == "XZY":
        R = torch.bmm(Ry, torch.bmm(Rz, Rx))
    elif convention == "YXZ":
        R = torch.bmm(Rz, torch.bmm(Rx, Ry))
    elif convention == "YZX":
        R = torch.bmm(Rx, torch.bmm(Rz, Ry))
    elif convention == "ZXY":
        R = torch.bmm(Ry, torch.bmm(Rx, Rz))
    elif convention == "ZYX":
        R = torch.bmm(Rx, torch.bmm(Ry, Rz))
    else:
        raise ValueError(f"Unsupported convention: {convention}")

    if batch_size == 1:
        R = R.squeeze(0)  # 返回 [3, 3]
    return R