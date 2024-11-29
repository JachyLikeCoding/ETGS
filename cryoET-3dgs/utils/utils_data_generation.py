# import numpy as np
# import torch
# import torch.nn.functional as F
# from scipy.optimize import minimize_scalar
# from scipy.spatial.transform import Rotation as scipy_rot


# def getRotationMatrix(angles, order='ZYZ', degrees=True):
#     rr = scipy_rot.from_euler(order, angles.cpu().numpy(), degrees)
#     mat = rr.as_matrix()
#     rotationMatrix = mat
#     return rotationMatrix

# def rotated_t(input_tensor, rotation_matrix):
#     device_ = input_tensor.device
#     d, h, w = input_tensor.shape
#     input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
#     R_ = torch.zeros((1,3,4))
#     R_[0,0,2] = 1
#     R_[0,1,1] = 1
#     R_[0,2,0] = 1

#     grid = F.affine_grid(R_, size=(1,1,d,h,w), align_corners=False).to(device_)
#     rotated_3d_positions = torch.matmul(rotation_matrix, grid.view(-1,3).T).T.view(1,d,h,w,3)
#     tmp = torch.cat((rotated_3d_positions[:,:,:,:,2].view(1,d,h,w,1), rotated_3d_positions[:,:,:,:,1].view(1,d,h,w,1), rotated_3d_positions[:,:,:,:,0].view(1,d,h,w,1)),dim=4)
#     rotated_signal = F.grid_sample(input=input_tensor, grid=tmp, mode='bilinear',align_corners=False).squeeze(0).squeeze(0)

#     return rotated_signal.to(device_)