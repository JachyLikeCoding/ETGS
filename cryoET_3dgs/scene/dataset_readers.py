import os
import numpy as np
import torch
import cv2
import glob
import mrcfile
import json
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from skimage import exposure, filters
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
import mrcfile
from typing import NamedTuple
from pathlib import Path
from cryoET_3dgs.scene.gaussian_model import BasicPointCloud
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
    weight: float
    mask: np.array
    z_clip: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_tilts: list
    test_tilts: list
    mrc_path: str
    vol: torch.tensor


def fetchPly(path, z_clip=0, threshold=None, sample_ratio=0.1, percentile=95, white_background=False):
    with mrcfile.open(path, mode='r') as mrc:
        data = np.transpose(mrc.data.astype(np.float32), (1,2,0))
        nx, ny, nz = data.shape
        zmin = nz // 2 - z_clip // 2
        zmax = nz // 2 + z_clip // 2
        print('zclip: ', z_clip)
        print('zmin, zmax:', zmin, zmax)

        if z_clip > 0 and z_clip < nz // 2:
            data = data[:, :, zmin:zmax]
        print(f'裁剪后尺寸: {data.shape}')
        print(f'[debug] data size: {data.shape[0]*data.shape[1]*data.shape[2]}')
        data_norm = (data - data.min()) / (data.max() - data.min())

        print(f'[debug] data mean: {data_norm.mean():.3f}')
        print('[debug] fetchPly threshold: ', threshold)
        print(f'[debug] fetchPly sample_ratio: {sample_ratio}')
        print(f'[debug] fetchPly white_background: {white_background}')
        
        # if threshold is None or threshold < 0.05:
        if threshold is None or threshold < 0.0005:
            try:
                # 先尝试Otsu方法
                otsu_thresh = threshold_otsu(data_norm)
                quant_thresh = np.quantile(data_norm, percentile/100.0)
                threshold = min(otsu_thresh, quant_thresh)
                print(f'自适应阈值: Otsu({otsu_thresh:.3f}), {percentile}%分位数({quant_thresh:.3f}), threshold({threshold:.3f})')
            except:
                threshold = np.quantile(data_norm, percentile/100.0)
                print(f'使用分位数阈值: {threshold:.3f}')

        mask = data_norm > threshold
        coords = np.argwhere(mask)
        intensities = data_norm[mask]
        print(f'阈值后点数：{len(coords)}')
        
        # 基于强度的概率采样
        if 0 < sample_ratio < 1 and len(coords) > 0:
            prob = intensities / intensities.sum()
            sample_size = int(len(coords) * sample_ratio)
            if sample_size > 2000000:
                sample_size = 2000000
            indices = np.random.choice(len(coords), sample_size, p=prob, replace=False)
            coords = coords[indices]
            intensities = intensities[indices]
            print(f'采样后点数: {len(coords)}')

        # random采样以使点云稀疏
        # num_points = positions.shape[0]
        # num_samples = int(num_points * sample_ratio)  # 根据 sample_ratio 计算采样点的数量
        # if num_samples > 0 and num_samples < num_points:
        #     sampled_indices = np.random.choice(num_points, num_samples, replace=False)
        #     positions = positions[sampled_indices]
        #     intensities = intensities[sampled_indices]    

        return BasicPointCloud(points=coords, intensities=intensities)
    


def readWbpInfo(path, images, eval, data, if_whitebg=False, if_align=True, if_render=False, llffhold=8):
    # read point clouds from WBP results
    reading_dir = "images" if images == None else images
    
    if if_render:
        tilt_files = glob.glob(os.path.join(path, 'render/*.tlt'))
    else:
        if if_align:
            # 查找 .aln 文件
            tilt_files = glob.glob(os.path.join(path, 'align/*.aln'))
        else:
            tilt_files = (
                glob.glob(os.path.join(path, '*.rawtlt')) +  # 查找所有.rawtlt文件
                glob.glob(os.path.join(path, '*.tlt'))       # 同时查找所有.tlt文件
            )
        
    tiltstack_file = glob.glob(os.path.join(path, 'mrc', '*.mrc'))

    if len(tilt_files) == 0:
        print(f"Error: No tilt file found. Expecting either .rawtlt or .tlt in {path}")
    elif len(tilt_files) > 1:
        # 显示具体冲突文件帮助调试
        print(f"Error: Found multiple tilt files ({len(tilt_files)}):")
        for f in tilt_files:
            print(f" - {os.path.basename(f)}")
        print("There should be exactly ONE tilt file in the directory")
    else:
        tilt_file = tilt_files[0]
        print(f"Tilt file detected: {os.path.basename(tilt_file)}")

    tilt_infos_unsorted = readTilts(os.path.join(path, tilt_file), tiltstack_file, data.axis_id, data.image_width, data.image_height, data.volume_z, if_render)
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
        if if_render:
            pcd = None
            vol_gt = None
        else:
            with mrcfile.open(mrc_path, mode='r') as mrc:
                mrc_data = np.transpose(mrc.data.astype(np.float32), (1,2,0)).copy()
                # lower = np.percentile(mrc_data, 0.5)
                # upper = np.percentile(mrc_data, 99.5)
                # mrc_data = np.clip(mrc_data, lower, upper)
                mrc_data = (mrc_data - mrc_data.min()) / (mrc_data.max() - mrc_data.min())
            vol_gt = torch.from_numpy(mrc_data).float().cuda()
            pcd = fetchPly(mrc_path, z_clip=data.volume_z, threshold=data.intensity_threshold, sample_ratio=data.sample_ratio, percentile=99, white_background=if_whitebg)
    elif len(mrc_files) > 1:
        pcd = None
        print("More than one mrc volume file, please check it out.")
    else:
        pcd = None
        print("No mrc volume file.")
    

    # 组装场景信息，包括点云、训练tilts、测试tilts、场景归一化参数和点云文件路径
    scene_info = SceneInfo(point_cloud=pcd,
                           train_tilts=train_tilt_infos,
                           test_tilts=test_tilt_infos,
                           mrc_path=mrc_path,
                           vol=vol_gt)

    return scene_info


def denoise_image(image):
    # 转为灰度图像
    image_gray = np.array(image.convert('L'))
    # 使用高斯滤波进行去噪
    image_denoised = cv2.GaussianBlur(image_gray, (5, 5), 1.0)
    return Image.fromarray(image_denoised)

# 归一化图像
def normalize_image(image):
    image_np = np.array(image)
    image_min, image_max = image_np.min(), image_np.max()
    normalized_image = (image_np - image_min) / (image_max - image_min)  # 归一化到 [0, 1] 范围
    return Image.fromarray((normalized_image * 255).astype(np.uint8))  # 转回 8 位图像

# CLAHE对比度增强
def enhance_contrast(image):
    image_np = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_np)
    return Image.fromarray(image_clahe)

# 预处理整合：去噪、归一化、增强等
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # 先去噪
    image = denoise_image(image)
    # 然后归一化
    image = normalize_image(image)
    # 再进行对比度增强
    image = enhance_contrast(image)
    return image


def invert_image(image):
    inverted_image = ImageOps.invert(image)
    return inverted_image


def readTilts(tilt_filepath, mrc_filepath=None, axis_id=0, width=1024, height=1024, zclip=0, if_render=False):
    print('[debug] rotate axis id: ', axis_id)
    print('[debug] mrc_filepath: ', mrc_filepath)
    tilt_axises = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

    #读取倾斜角度文件，支持两种格式：
    # 1. 纯角度文件（.tlt）：每行一个数字，如 "-60.0"
    # 2. Aretomo对齐文件（.aln）：含TILT列，如 "0 -5.0842 1.0 ... -58.90"
    
    if tilt_filepath.endswith(('.tlt', '.rawtlt')):
        with open(tilt_filepath, 'r') as f:
            tilt_angles = [float(line.strip()) for line in f]
    else: # .aln file
        with open(tilt_filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]  # 读取非空行

        # 尝试作为.tlt文件读取（所有行都是数字）
        try:
            tilt_angles = [float(line) for line in lines if not line.startswith('#')]
        except ValueError:
            pass  # 不是纯数字文件，继续尝试解析.aln格式

        # 尝试作为.aln文件读取（查找TILT列）
        tilt_angles = []
        tilt_col_index = None
        
        for line in lines:
            if line.startswith('#'):
                if "TILT" in line and not tilt_col_index:
                    header = line[1:].split()  # 去掉#后分割列名
                    tilt_col_index = header.index("TILT")
                continue
            
            parts = line.split()
            if tilt_col_index is not None and len(parts) > tilt_col_index:
                tilt_angles.append(float(parts[tilt_col_index]))
    
    if tilt_angles:
        print('[debug] 读取到的倾斜角度:', tilt_angles)
    else:
        raise ValueError("无法识别文件格式：既不是纯角度.tlt文件, 也不是含TILT列的.aln文件")
    
    # 初始化用于存储倾斜序列信息的列表
    tilts_infos = []
    
    folder = os.path.dirname(tilt_filepath)

    if if_render:
        for i in range(len(tilt_angles)):
            tilt = tilt_angles[i]
            _R, _T, _mask = cal_R_T_mask(tilt, width, height, tilt_axis=tilt_axises[axis_id])
            # 计算权重，基于倾斜角度
            weight = calculate_weight_from_tilt(tilt)
            tilt_info = TiltInfo(uid=i, tilt=tilt, R=_R, T=_T, image=None, image_path=None, image_name=f"slice_{i:02d}", width=width, height=height, weight=weight, mask=_mask, z_clip=zclip)
            tilts_infos.append(tilt_info)
    else:
        if mrc_filepath:
            with mrcfile.open(mrc_filepath[0], mode='r', permissive=True) as mrc:
                # mrc_data = mrc.data
                mrc_data = mrc.data.astype(np.float32)
                n_slices = mrc_data.shape[0]
                global_min, global_max = np.min(mrc_data), np.max(mrc_data)
                
                if len(tilt_angles) != n_slices:
                    raise ValueError(f" The number of tilts ({len(tilt_angles)}) is not equal to number of projections ({n_slices}.)")
                for i in range(n_slices):
                    slice_data = mrc_data[i]
                    normalized = (slice_data - global_min) / (global_max - global_min)
                    if len(tilt_angles) > 50: # shrec
                        lower = np.percentile(slice_data, 0.5)
                        upper = np.percentile(slice_data, 99.5)
                        clipped = np.clip(slice_data, lower, upper)
                        normalized = (clipped - lower) / (upper - lower + 1e-8) * 255
                    elif len(tilt_angles) < 40: # 10453
                        lower = np.percentile(slice_data, 0.5)
                        upper = np.percentile(slice_data, 99.5)
                        clipped = np.clip(slice_data, lower, upper)
                        data_denoised = filters.median(clipped)  # 中值滤波降噪
                        normalized = (data_denoised - lower) / (upper - lower + 1e-8) * 255
                    else: # empiar
                        lower = np.percentile(slice_data, 0.1)
                        upper = np.percentile(slice_data, 99.9)
                        clipped = np.clip(slice_data, lower, upper)
                        # normalized = exposure.equalize_adapthist(clipped)
                        data_denoised = filters.median(clipped)  # 中值滤波降噪
                        normalized = (data_denoised - lower) / (upper - lower + 1e-8) * 255
                        
                    image = Image.fromarray(normalized.T.astype(np.uint8))
                    image = image.resize((width, height), Image.Resampling.LANCZOS)

                    tilt = tilt_angles[i]
                    _R, _T, _mask = cal_R_T_mask(tilt, width, height, tilt_axis=tilt_axises[axis_id])
                    # 基于倾斜角度计算权重
                    weight = calculate_weight_from_tilt(tilt)
                    # save tilt series information
                    tilt_info = TiltInfo(uid=i, tilt=tilt, R=_R, T=_T, image=image, image_path=mrc_filepath, image_name=f"slice_{i:02d}", width=width, height=height, weight=weight, mask=_mask, z_clip=zclip)
                    tilts_infos.append(tilt_info)
        else:
            image_files = sorted([f for f in os.listdir(os.path.join(folder, 'images')) if f.endswith('.png')])
            if len(image_files) > 0:
                i = 0
                with open(tilt_filepath, 'r') as f:
                    for line in f:
                        tilt = float(line.strip())
                        # 根据图像索引生成对应的文件名
                        image_name = image_files[i]
                        image_path = os.path.join(folder, 'images', image_name)
                        image = Image.open(image_path)
                        
                        # 图像预处理 - 降噪与归一化
                        # image = preprocess_image(image)

                        # tomo dataset
                        # image = image.resize((width, height))
                        # image = image.rotate(-90)
                        
                        # 10164 dataset
                        # image = image.resize((width, height))
                        # image = image.transpose(0)
                        
                        # 10643 dataset
                        # image = preprocess_image(image)
                        # image = image.resize((width, height))
                        # image = image.transpose(0)
                        
                        # # 10453 dataset
                        # image = image.resize((1024, 1024))
                        # image = image.transpose(0)
                        
                        # simulate dataset
                        # print('[debug] Initial image size: ', image.size)
                        # image = image.resize((width, height))
                        # image = image.rotate(90)
                        # image = image.transpose(1)

                        # shrec dataset
                        image = image.resize((width, height))
                        image = image.rotate(90)
                        # 白色背景
                        image = invert_image(image)
                        
                        _R, _T, _mask = cal_R_T_mask(tilt, width, height, tilt_axis=tilt_axises[axis_id])
                        # 计算权重，基于倾斜角度
                        weight = calculate_weight_from_tilt(tilt)  # 使用倾斜角度计算权重

                        # save tilt series information
                        tilt_info = TiltInfo(uid=i, tilt=tilt, R=_R, T=_T, image=image, image_path=image_path, image_name=image_name, width=width, height=height, weight=weight, mask=_mask, z_clip=zclip)
                        tilts_infos.append(tilt_info)
                        i += 1

    for tiltinfo in tilts_infos:
        print(tiltinfo)
        print()

    return tilts_infos


def calculate_weight_from_tilt(tilt):
    # 用高斯函数计算权重
    # weight = np.exp(-tilt ** 2 / (2 * 10 ** 2))
    weight = -1/3600 * tilt ** 2 + 1.5
    weight = np.clip(weight, 0.5, 1.5)
    # weight = 1
    return weight


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


def cal_R_T_mask(tilt_angle, width, height, tilt_axis):
    tilt_angle_radians = np.radians(tilt_angle)

    R = cv2.Rodrigues(tilt_axis * tilt_angle_radians)[0]

    # 平移向量 T 应该为零向量
    T = np.array([0, 0, 0])

    factor = np.cos(np.radians(tilt_angle))
    
    mask = np.zeros((width, height))
    if tilt_axis[0] == 1:
        effective_height = int(height * factor)
        left = (height - effective_height)//2
        right = left + effective_height
        mask[:, left:right] = 1.0
    elif tilt_axis[1]==1:
        effective_width = int(width * factor)
        left = (width - effective_width)//2
        right = left + effective_width
        mask[left:right, :] = 1.0

    return R, T, mask



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
        print(mrcstack[0])
        for i, image in enumerate(mrcstack):
            # 将图像数据转换为 8-bit 格式
            image_8bit = (image - image.min()) / (image.max() - image.min()) * 255
            image_8bit = image_8bit.astype('uint8')
            # 上下翻转图像
            flipped_image = cv2.flip(image_8bit, 0)
            # denoised_image = cv2.GaussianBlur(flipped_image, (5, 5), 1.0)
            # normalized_image = (denoised_image - denoised_image.min()) / (denoised_image.max() - denoised_image.min()) * 255
            # normalized_image = normalized_image.astype(np.uint8)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # enhanced_image = clahe.apply(normalized_image)
            preprocessed_image = cv2.equalizeHist(flipped_image)
            # 如果 preprocessed_image 是 PIL.Image，转换为 numpy.ndarray
            if isinstance(preprocessed_image, Image.Image):
                preprocessed_image = np.array(preprocessed_image)
            # 写入PNG图像
            print(f'/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/data/{dataset}/TS_099/images/{prefix}_{i}.png')
            # cv2.imwrite(f'/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/data/{dataset}/TS_099/images/{prefix}_{i}.png', flipped_image)
            cv2.imwrite(f'/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/data/{dataset}/TS_099/images/{prefix}_{i}.png', preprocessed_image)


def test_visualize_point_cloud():
    mrcfile_path = "/home/feng/Desktop/cryoET-reconstruction/Implicit-Cryo-Electron-Tomography/results/tkiuv_tomo2_L1G1-dose_filt/training/V_est_final.mrc"
    point_cloud = fetchPly(mrcfile_path, 100, 0.3, 0.1)
    print("Number of points:", len(point_cloud.points))
    print("Number of densities:", len(point_cloud.intensities))
    visualize_point_cloud(point_cloud, 0.1)


if __name__ == '__main__':
    # mrcfilepath = '/home/feng/Desktop/cryoET-reconstruction/cryoET-3DGS/data/tomo/tomo2_L1G1-dose_filt.mrc'  # 替换为实际的MRC文件路径
    # mrcfilepath = '/home/feng/Desktop/cryoET-reconstruction/cryoET-3DGS/data/10643/b2tilt40.mrc'
    # mrcfilepath = '/media/feng/2480CDB880CD90AA/cryoET-reconstruction/cryoET-3DGS/data/10453/TS_099/TS_099.mrc'
    # save_mrc_as_png('10453', mrcfilepath)
    
    # tiltfilepath = '/home/feng/Desktop/cryoET-reconstruction/cryoET-3DGS/data/tomo/tomo2_L1G1-dose_filt.rawtlt'
    # readTilts(tiltfilepath)

    # Initial coordinates in the sample coordinate system
    # initial_coordinates = np.array([70, 100, 0])
    # R, T = cal_R_T(60.0, tilt_axis=np.array([1, 0, 0]))
    # print(R, T)

    # # Apply the rotation to the initial coordinates
    # new_coordinates = np.dot(R, initial_coordinates) + T
    # new_coordinates_int = new_coordinates.astype(int)
    # print("New coordinates after tilting 30 degrees:")
    # print(new_coordinates)
    # 使用示例

    R, T, mask = cal_R_T_mask(30, 720, 511, np.array([1, 0, 0]))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.savefig('mask.png')
    plt.close()