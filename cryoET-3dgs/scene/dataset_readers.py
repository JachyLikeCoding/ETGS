import os,sys
import numpy as np
import torch
import cv2
import glob
import json
from PIL import Image
import mrcfile
from typing import NamedTuple
from pathlib import Path
from scene.gaussian_model import BasicPointCloud
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class TiltInfo(NamedTuple):
    uid: int
    tilt: float
    R: np.array
    T: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int



class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_tilts: list
    test_tilts: list
    mrc_path: str



def fetchPly(path, z_clip, threshold, sample_ratio=0.1):
    with mrcfile.open(path, mode='r') as mrc:
        data = mrc.data.astype(np.float32)
        data = np.transpose(data,(1,2,0))
        shape = data.shape
        nx, ny, nz = shape
        print('[debug] fetchPly')
        print('nx,ny,nz:', nx,ny,nz)

        zmin = int(nz/2) - int(z_clip/2)
        zmax = int(nz/2) + int(z_clip/2)
        print(zmin, zmax)

        if z_clip > 0 and z_clip < nz/2:
            data = data[:, :, int(zmin):int(zmax)]
   
        shape = data.shape
        nx, ny, nz = shape
        print(nx,ny,nz)

        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)

        xx, yy, zz = np.meshgrid(x,y,z,indexing='ij')
        positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        intensities = data.ravel()

        # 归一化密度值到0到1的范围内
        intensities /= intensities.max()
        mask = intensities > threshold

        positions = positions[mask]
        intensities = intensities[mask]

        # 采样以使点云稀疏
        num_points = positions.shape[0]
        num_samples = int(num_points * sample_ratio)  # 根据 sample_ratio 计算采样点的数量
        if num_samples > 0 and num_samples < num_points:
            sampled_indices = np.random.choice(num_points, num_samples, replace=False)
            positions = positions[sampled_indices]
            intensities = intensities[sampled_indices]    

        # 将 Z 坐标范围调整到 [0, z_clip]
        positions[:, 2] = (positions[:, 2] - positions[:, 2].min()) * (z_clip / (positions[:, 2].max() - positions[:, 2].min()))

        return BasicPointCloud(points=positions, intensities=intensities)
    
    

# 读取FBP重建结果
def readFbpSceneInfo(path, images, eval, llffhold=8):
    # read point clouds from FBP results
    '''TODO: change for FBP results''' 

    reading_dir = "images" if images == None else images

    tilt_files = glob.glob(os.path.join(path, '*.rawtlt'))

    if len(tilt_files) != 1:
        print("Error: There should be exactly one .rawtlt file in the directory.")
    else:
        tilt_file = tilt_files[0]
        print("Tilt file found:", tilt_file)

    tilt_infos_unsorted = readTilts(os.path.join(path, tilt_file))
    # 根据图片名称对倾斜序列信息进行排序，以保证顺序一致性
    tilt_infos = sorted(tilt_infos_unsorted.copy(), key=lambda x : x.image_name)

    # 根据是否为eval模式将倾斜序列分为训练集和测试集。如果是eval模式，根据llffhold参数间隔选择测试相机
    if eval:
        train_tilt_infos = [c for idx,c in enumerate(tilt_infos) if idx % llffhold != 0]
        test_tilt_infos = [c for idx,c in enumerate(tilt_infos) if idx % llffhold == 0]
    else:
        train_tilt_infos = tilt_infos
        test_tilt_infos = []

    # 获取文件夹中的所有 MRC 文件
    mrc_files = [f for f in os.listdir(os.path.join(path, 'sparse')) if f.endswith('.mrc')]

    # 检查是否存在唯一的 MRC 文件
    if len(mrc_files) == 1:
        mrc_path = os.path.join(path, 'sparse', mrc_files[0])
        pcd = fetchPly(mrc_path)
    elif len(mrc_files) > 1:
        pcd = None
        print("More than one mrc volume file, please check it out.")
    else:
        pcd = None
        print("No mrc volume file.")

    
    # 组装场景信息，包括点云、训练tilts、测试tilts、场景归一化参数和点云文件路径
    scene_info = SceneInfo(point_cloud=pcd,
                           train_tilt_infos=train_tilt_infos,
                           test_tilt_infos=test_tilt_infos,
                           mrc_path=mrc_path)

    return scene_info

def denoise_image(image):
    # 转为灰度图像
    image_gray = np.array(image.convert('L'))
    # 使用高斯滤波进行去噪
    image_denoised = cv2.GaussianBlur(image_gray, (5, 5), 1.0)
    return Image.fromarray(image_denoised)

# 示例：归一化图像
def normalize_image(image):
    image_np = np.array(image)
    image_min, image_max = image_np.min(), image_np.max()
    normalized_image = (image_np - image_min) / (image_max - image_min)  # 归一化到 [0, 1] 范围
    return Image.fromarray((normalized_image * 255).astype(np.uint8))  # 转回 8 位图像

# 示例：CLAHE对比度增强
def enhance_contrast(image):
    image_np = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_np)
    return Image.fromarray(image_clahe)

# 预处理整合：去噪、归一化、增强等
def preprocess_image(image):
    # 先去噪
    image = denoise_image(image)
    # 然后归一化
    image = normalize_image(image)
    # 再进行对比度增强
    image = enhance_contrast(image)
    return image


def readTilts(filepath, axis_id=0):
    print('[debug] rotatie axis id: ', axis_id)
    tilt_axises = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    prefix = os.path.splitext(os.path.basename(filepath))[0]
    # 初始化用于存储倾斜序列信息的列表
    tilts_infos = []
    folder = os.path.dirname(filepath)
    image_files = sorted([f for f in os.listdir(os.path.join(folder, 'images')) if f.endswith('.png')])

    i = 0
    with open(filepath, 'r') as f:
        for line in f:
            tilt = float(line.strip())
            # 根据图像索引生成对应的文件名
            # image_name = f'{prefix}_{i}.png'  
            image_name = image_files[i]
            image_path = os.path.join(folder, 'images', image_name)
            image = Image.open(image_path)
            
            
            # 图像预处理 - 降噪与归一化
            image = preprocess_image(image)
            
            # tomo dataset
            # image = image.resize((1024, 1024))
            # image = image.rotate(-90)
            
            # 10164 dataset
            image = image.resize((1024, 1024))
            image = image.transpose(0)

            width, height = image.size
            _R, _T = cal_R_T(tilt, tilt_axis=tilt_axises[axis_id])
            tilt_info = TiltInfo(uid=i, tilt=tilt, R=_R, T=_T, image=image, image_path=image_path, image_name=image_name, width=width, height=height)
            tilts_infos.append(tilt_info)
            i += 1

    for tiltinfo in tilts_infos:
        print(tiltinfo)
        print()

    return tilts_infos



# def cal_R_T(tilt_angle, tilt_axis=np.array([1, 0, 0])):
#     '''
#     假设样品中心位于原点位置, 样品是绕X轴旋转, 电子束沿着Z轴负方向。
#     旋转角度为 tilt_angle
#     由于样品中心位于原点, 平移向量T应该为零向量
#     '''
#     tilt_angle_radians = np.radians(tilt_angle)
#     R = np.array([[1, 0, 0],
#                   [0, np.cos(tilt_angle_radians), -np.sin(tilt_angle_radians)],
#                   [0, np.sin(tilt_angle_radians), np.cos(tilt_angle_radians)]])

#     # T = np.array([0, focus * np.sin(tilt_angle_radians), focus * np.cos(tilt_angle_radians)])
#     T = np.array([0, 0, 0])
#     return R, T

def cal_R_T(tilt_angle, tilt_axis=np.array([0, 1, 0])):
    '''
    假设样品中心位于原点位置, 样品是绕Y轴旋转, 电子束沿着Z轴负方向。
    旋转角度为 tilt_angle
    由于样品中心位于原点, 平移向量T应该为零向量
    '''
    tilt_angle_radians = np.radians(tilt_angle)
    # 使用旋转轴tilt_axis来生成旋转矩阵
    # R = np.array([[np.cos(tilt_angle_radians), 0, np.sin(tilt_angle_radians)],
    #               [0, 1, 0],
    #               [-np.sin(tilt_angle_radians), 0, np.cos(tilt_angle_radians)]])
    R = cv2.Rodrigues(tilt_axis * tilt_angle_radians)[0]

    # 平移向量 T 应该为零向量
    T = np.array([0, 0, 0])
    return R, T


def readRelionInfo(path, images, eval, data, llffhold=8):
    reading_dir = "images" if images == None else images
    tilt_files = glob.glob(os.path.join(path, '*.rawtlt'))

    if len(tilt_files) != 1:
        print("Error: There should be exactly one .rawtlt file in the directory.")
    else:
        tilt_file = tilt_files[0]
        print("Tilt file found:", tilt_file)

    tilt_infos_unsorted = readTilts(os.path.join(path, tilt_file), data.axis_id)
    tilt_infos = sorted(tilt_infos_unsorted.copy(), key=lambda x : x.image_name)

    if eval:
        train_tilt_infos = [c for idx,c in enumerate(tilt_infos) if idx % llffhold != 0]
        test_tilt_infos = [c for idx,c in enumerate(tilt_infos) if idx % llffhold == 0]
    else:
        train_tilt_infos = tilt_infos
        test_tilt_infos = []


    # 获取文件夹中的所有 MRC 文件
    mrc_files = [f for f in os.listdir(os.path.join(path, 'sparse')) if f.endswith('.mrc')]

    # 检查是否存在唯一的 MRC 文件
    if len(mrc_files) == 1:
        mrc_path = os.path.join(path, 'sparse', mrc_files[0])
    elif len(mrc_files) > 1:
        pcd = None
        print("More than one mrc volume file, please check it out.")
    else:
        pcd = None
        print("No mrc volume file.")

    try:
        pcd = fetchPly(mrc_path, z_clip=data.volume_z, threshold=data.intensity_threshold, sample_ratio=data.sample_ratio)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_tilts=train_tilt_infos,
                           test_tilts=test_tilt_infos,
                           mrc_path=mrc_path)

    return scene_info



sceneLoadTypeCallbacks = {
    "fbp": readFbpSceneInfo,
    "relion": readRelionInfo
}



def visualize_point_cloud(point_cloud, sampling_rate=0.01):
    """
    将点云可视化出来。
    
    Parameters:
        point_cloud (BasicPointCloud): BasicPointCloud对象,包含坐标值和密度值。
        sampling_rate (float): 下采样率 表示保留的点的比例,默认为 0.1。
    """
    num_points = len(point_cloud.points)
    print('points number: ', num_points)

    sampled_indices = np.random.choice(num_points, size=int(num_points * sampling_rate), replace=False)
    sampled_points = point_cloud.points[sampled_indices]
    sampled_intensities = point_cloud.intensities[sampled_indices]
    print('sample points: ', len(sampled_points))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c=sampled_intensities, cmap='gray')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # 设置每个轴的比例
    scatter = ax.set_box_aspect([np.ptp(sampled_points[:, 0]), np.ptp(sampled_points[:, 1]), np.ptp(sampled_points[:, 2])])

    plt.savefig('init.png')


def readMrcStack(filepath):
    perfix = os.path.basename(filepath)
    with mrcfile.open(filepath, mode='r') as mrc:
        mrcstack = mrc.data
        for i, image in enumerate(mrcstack):
            import cv2
            cv2.imwrite(image, f'{perfix}_{i}.png')



def save_mrc_as_png(dataset, filepath):
    prefix = os.path.splitext(os.path.basename(filepath))[0]
    with mrcfile.open(filepath, mode='r') as mrc:
        mrcstack = mrc.data
        for i, image in enumerate(mrcstack):
            # 将图像数据转换为 8-bit 格式
            image_8bit = (image - image.min()) / (image.max() - image.min()) * 255
            image_8bit = image_8bit.astype('uint8')
            # 上下翻转图像
            flipped_image = cv2.flip(image_8bit, 0)
            # 写入PNG图像
            cv2.imwrite(f'/home/feng/Desktop/cryoET-reconstruction/cryoET-3DGS/data/{dataset}/images/{prefix}_{i}.png', flipped_image)



def test_visualize_point_cloud():
    mrcfile_path = "/home/feng/Desktop/cryoET-reconstruction/Implicit-Cryo-Electron-Tomography/results/tkiuv_tomo2_L1G1-dose_filt/training/V_est_final.mrc"
    point_cloud = fetchPly(mrcfile_path, 100, 0.3, 0.1)
    print("Number of points:", len(point_cloud.points))
    print("Number of densities:", len(point_cloud.intensities))
    visualize_point_cloud(point_cloud, 0.1)



if __name__ == '__main__':
    # mrcfilepath = '/home/feng/Desktop/cryoET-reconstruction/cryoET-3DGS/data/tomo/tomo2_L1G1-dose_filt.mrc'  # 替换为实际的MRC文件路径
    # mrcfilepath = '/home/feng/Desktop/cryoET-reconstruction/cryoET-3DGS/data/10643/b2tilt40.mrc'
    # save_mrc_as_png('10643', mrcfilepath)
    
    # tiltfilepath = '/home/feng/Desktop/cryoET-reconstruction/cryoET-3DGS/data/tomo/tomo2_L1G1-dose_filt.rawtlt'
    # readTilts(tiltfilepath)


    # Initial coordinates in the sample coordinate system
    initial_coordinates = np.array([70, 100, 0])
    R, T = cal_R_T(60.0, tilt_axis=np.array([1, 0, 0]))
    print(R, T)

    # Apply the rotation to the initial coordinates
    new_coordinates = np.dot(R, initial_coordinates) + T
    new_coordinates_int = new_coordinates.astype(int)
    print("New coordinates after tilting 30 degrees:")
    print(new_coordinates)