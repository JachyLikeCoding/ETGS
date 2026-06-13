import torch
import math
from torch import nn
import numpy as np
from cryoET_3dgs.utils.graphics_utils import getSample2View, getOrthographicProjectionMatrix, euler_angles_to_matrix
from cryoET_3dgs.scene.dataset_readers import cal_R_T
from cryoET_3dgs.scene.gaussian_model import GaussianModel_cryoET


# 实验中样本是在旋转的，为了建立相机坐标系，假设样本不动相机坐标系在动。
class Tilt(nn.Module):
    def __init__(self, tilt_id, mask, R, T, width, height, image, 
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, zmax=0.0, weight = 1.0, data_device="cuda", tilt_no=None, learn_residual=True):
        super(Tilt, self).__init__()
        self.uid = uid
        self.tilt_id = tilt_id
        self.mask = mask
        self.T = T
        self.R = R
        self.image_name = image_name
        self.tilt_no = tilt_no
        self.learn_residual = learn_residual
        
        if self.learn_residual:
            self.delta_rot = nn.Parameter(torch.zeros(3, dtype=torch.float32)) # [theta_x, theta_y, theta_z]
            self.delta_trans = nn.Parameter(torch.zeros(3, dtype=torch.float32)) # 平移残差 dx, dy, dz

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.zmin = -zmax
        self.zmax =  zmax

        # 平移和缩放
        self.trans = trans
        self.scale = scale
        
        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device) # 原图限制到0-1范围内了
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        else:
            self.image_width = width
            self.image_height = height
        
        # 从样品坐标系变换到相机坐标系
        self.sample_view_transform = torch.tensor(getSample2View(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        # 获取相机坐标系到NDC坐标系的投影矩阵
        self.projection_matrix = getOrthographicProjectionMatrix(znear=self.zmin, zfar=self.zmax).transpose(0, 1).to(self.data_device)
        # 两矩阵一相乘，计算完整的投影变换矩阵
        self.full_proj_transform = (self.sample_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.weight = weight
        
        # 加入变形场
        self.rigid_deformation = torch.zeros((1,6), device=self.data_device) # 3D旋转+3D平移
        self.non_rigid_deformation = {} # 非刚性变形，针对每个高斯点的局部变形场


    def get_actual_transform(self):
        R_theory = self.R
        if self.learn_residual:
            delta_rot_mat = euler_angles_to_matrix(self.delta_rot, convention="XYZ")
            R_actual = torch.matmul(delta_rot_mat, R_theory)
            T_actual = self.T + self.delta_trans
        else:
            R_actual = R_theory
            T_actual = self.T
        return R_actual, T_actual


    def update_transform_matrix(self):
        """更新最终的投影变换矩阵（需在每次参数更新后调用）"""
        R_actual, T_actual = self.get_actual_transform()
        
        # 重新计算包含残差的变换矩阵
        self.sample_view_transform = torch.tensor(getSample2View(R_actual, T_actual, self.trans, self.scale)).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = self.sample_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)


class MiniTilt:
    def __init__(self, width, height, znear, zfar, sample_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = sample_view_transform
        self.full_proj_transform = full_proj_transform


class CustomTilt:
    def __init__(self, width, height, tilt_angle, trans=np.array([0.0, 0.0, 0.0]), tilt_axis_id=0, scale=1.0, znear=0, zfar=0):
        self.image_width = width
        self.image_height = height    
        self.zmin = znear
        self.zmax = zfar
        if tilt_axis_id == 0:
            tilt_axis = np.array([1, 0, 0])
        elif tilt_axis_id == 1:
            tilt_axis = np.array([0, 1, 0])
        elif tilt_axis_id == 2:
            tilt_axis = np.array([0, 0, 1])
        R, T = cal_R_T(tilt_angle, tilt_axis)
        self.T = T
        self.R = R
        self.trans = trans
        self.scale = scale
        # 从样品坐标系变换到相机坐标系
        self.sample_view_transform = torch.tensor(getSample2View(R, T, trans, scale)).transpose(0, 1).cuda()
        # 获取相机坐标系到NDC坐标系的投影矩阵
        self.projection_matrix = getOrthographicProjectionMatrix(znear, zfar).transpose(0, 1).cuda()
        # 两矩阵一相乘，计算完整的投影变换矩阵
        self.full_proj_transform = (self.sample_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = -self.full_proj_transform[:3, -1]