# import numpy as np
# import torch
# from utils.utils_data_generation import getRotationMatrix, rotated_t
# import mrcfile
# import os,sys


# def SNR(x, xhat):
#     """Returns SNR of xhat wrt to ground truth image x."""
#     diff = x - xhat

#     return -20 * np.log10(np.linalg.norm(diff) / np.linalg.norm(x))


# def backprojection(projections, angles, weightType=0, ignoreAxis=2, polinomialOrder=2, order='X', degrees=True):

#     n = projections.shape[1]
#     print('projection shape: ', projections.shape)
#     angleMatrixTorch = getRotationMatrix(angles, order=order, degrees=degrees)
#     angleMatrixTorch = torch.tensor(angleMatrixTorch).to(projections.device).type_as(projections)
    
#     projectionTorch = projections
#     reconVolume = torch.zeros(n,n,n).to(projections.device).type_as(projections)

#     if(weightType==0):
#         weights = torch.ones((angleMatrixTorch.shape[0],1)).to(projections.device)

#     weights = weights/torch.max(abs(weights))

#     for angleMatrix, projection, weight in zip(angleMatrix,  projectionTorch, weights):
#         projection = projection.type(torch.FloatTensor).to(projectionTorch.device).type_as(projections)
#         backprojectedVolume = projection.unsqueeze(2).repeat(1,1,projection.shape[0])
#         backprojectedVolume = rotated_t(backprojectedVolume, angleMatrix.T)
#         reconVolume = reconVolume + weight * backprojectedVolume

#     return reconVolume


# def load_data():
#     projections_noisy = torch.Tensor(np.float32(mrcfile.open(os.path.join("/home/feng/Desktop/cryoET-reconstruction/Implicit-Cryo-Electron-Tomography/datasets/tkiuv/","tomo2_L1G1.mrc"),permissive=True).data)).type(torch.float).to(device).clone()
#     projections_noisy = projections_noisy/torch.abs(projections_noisy).max() # make sure that values to predict are between -1 and 1
    
#     angles = np.linspace(-60, 60, 60)
#     angles = torch.tensor(angles).type(torch.float).to(device)
#     return projections_noisy, angles

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     projections,angles = load_data()
#     reconVolume = backprojection(projections=projections, angles=angles)
#     out = mrcfile.new(f"BP.mrc",overwrite=True)
#     out.close() 

import numpy as np
import torch
from utils.utils_data_generation import getRotationMatrix, rotated_t
import mrcfile
import os,sys


def SNR(x, xhat):
    """Returns SNR of xhat wrt to ground truth image x."""
    diff = x - xhat

    return -20 * np.log10(np.linalg.norm(diff) / np.linalg.norm(x))


def backprojection(projections, angles, weightType=0, ignoreAxis=2, polinomialOrder=2, order='X', degrees=True):

    n = int(projections.shape[1]*0.2)
    print('projection shape: ', projections.shape)
    angleMatrixTorch = getRotationMatrix(angles, order=order, degrees=degrees)
    angleMatrixTorch = torch.tensor(angleMatrixTorch).to(projections.device).type_as(projections)
    
    projectionTorch = projections
    reconVolume = torch.zeros(n,n,n).to(projections.device).type_as(projections)

    if(weightType==0):
        weights = torch.ones((angleMatrixTorch.shape[0],1)).to(projections.device)

    weights = weights/torch.max(abs(weights))

    for angleMatrix, projection, weight in zip(angleMatrixTorch,  projectionTorch, weights):
        projection = projection.type(torch.FloatTensor).to(projectionTorch.device).type_as(projections)
        
        # 降采样
        projection = torch.nn.functional.interpolate(projection.unsqueeze(0).unsqueeze(0), scale_factor=0.2, mode='bilinear').squeeze()
        
        backprojectedVolume = projection.unsqueeze(2).repeat(1,1,projection.shape[0])
        backprojectedVolume = rotated_t(backprojectedVolume, angleMatrix.T)
        reconVolume = reconVolume + weight * backprojectedVolume

    return reconVolume


def load_data():
    projections_noisy = torch.Tensor(np.float32(mrcfile.open(os.path.join("/home/feng/Desktop/cryoET-reconstruction/Implicit-Cryo-Electron-Tomography/datasets/tkiuv/","tomo2_L1G1.mrc"),permissive=True).data)).type(torch.float).to(device).clone()
    projections_noisy = projections_noisy/torch.abs(projections_noisy).max() # make sure that values to predict are between -1 and 1
    
    angles = np.linspace(-60, 60, 60)
    angles = torch.tensor(angles).type(torch.float).to(device)
    return projections_noisy, angles

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projections,angles = load_data()
    reconVolume = backprojection(projections=projections, angles=angles)
    out = mrcfile.new(f"BP.mrc",overwrite=True)
    out.close() 
