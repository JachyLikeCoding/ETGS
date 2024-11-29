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
    Rt[:3, 3] = (T + translate) * scale
    return np.float32(Rt)


def getOrthographicProjectionMatrix(znear, zfar):
    """
    Generate an orthographic projection matrix.
    """
    ortho_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],  # 保持x坐标不变
        [0.0, 1.0, 0.0, 0.0],  # 保持y坐标不变
        [0.0, 0.0, 2.0/(znear-zfar), (znear+zfar)/(znear-zfar)],  # z坐标映射到[-1, 1]区间
        [0.0, 0.0, 0.0, 1.0]   # 齐次坐标分量
    ])

    return torch.tensor(ortho_matrix, dtype=torch.float32)