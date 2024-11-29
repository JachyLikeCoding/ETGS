import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getSample2View, getOrthographicProjectionMatrix
from scene.dataset_readers import cal_R_T

# 在真实实验中，样本是在旋转的，但是为了建立相机坐标系，我们假设样本不动，相机坐标系在动。

class Tilt(nn.Module):
    def __init__(self, tilt_id, R, T, image, 
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        super(Tilt, self).__init__()

        self.uid = uid
        self.tilt_id = tilt_id
        self.T = T
        self.R = R
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.zmin = -512.0
        self.zmax =  512.0

        # 平移和缩放
        self.trans = trans
        self.scale = scale

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device) # 原图限制到0-1范围内了
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 从样品坐标系变换到相机坐标系
        self.sample_view_transform = torch.tensor(getSample2View(R, T, trans, scale)).transpose(0, 1).cuda()
        # 获取相机坐标系到NDC坐标系的投影矩阵
        self.projection_matrix = getOrthographicProjectionMatrix(znear=self.zmin, zfar=self.zmax).transpose(0, 1).cuda()
        # 两矩阵一相乘，计算完整的投影变换矩阵
        self.full_proj_transform = (self.sample_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        


class MiniTilt:
    def __init__(self, width, height, znear, zfar, sample_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = sample_view_transform
        self.full_proj_transform = full_proj_transform




class CustomTilt:
    def __init__(self, width, height, tilt_angle, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, znear=-512, zfar=512):
        self.image_width = width
        self.image_height = height    
        self.zmin = -512.0
        self.zmax =  512.0
        R, T = cal_R_T(tilt_angle, tilt_axis=np.array([1, 0, 0]))
        self.T = T
        self.R = R
        # 平移和缩放
        self.trans = trans
        self.scale = scale
        # 从样品坐标系变换到相机坐标系
        self.sample_view_transform = torch.tensor(getSample2View(R, T, trans, scale)).transpose(0, 1).cuda()
        # 获取相机坐标系到NDC坐标系的投影矩阵
        self.projection_matrix = getOrthographicProjectionMatrix(znear, zfar).transpose(0, 1).cuda()
        # 两矩阵一相乘，计算完整的投影变换矩阵
        self.full_proj_transform = (self.sample_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = -self.full_proj_transform[:3, -1]
