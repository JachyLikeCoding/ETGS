import os
import numpy as np
import random
import json

from scene.gaussian_model import GaussianModel_cryoET
from scene.dataset_readers import sceneLoadTypeCallbacks
from arguments import ModelParams, DatasetParams
from utils.system_utils import searchForMaxIteration
from utils.tilt_utils import tiltList_from_tiltInfos, tilt_to_JSON



# 数据加载
class Scene:
    """
    Scene 类用于管理场景的3D模型, 包括倾斜序列参数、点云数据和高斯模型的初始化和加载。
    """
    gaussians: GaussianModel_cryoET

    def __init__(self, args: ModelParams, data: DatasetParams, gaussians: GaussianModel_cryoET, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        初始化场景对象
        :param args: 包含模型路径和源路径等模型参数
        :param gaussians: 高斯模型对象,用于场景点的3D表示
        :param load_iteration: 指定加载模型的迭代次数，如果是-1则自动寻找最大迭代次数
        :param shuffle: 是否在训练前打乱倾斜序列列表
        :param resolution_scales:分辨率比例列表，用于处理不同分辨率的投影图像
        """
        self.model_path = args.model_path # 模型文件保存路径
        self.loaded_iter = None # 已加载的迭代次数
        self.gaussians = gaussians  # 高斯模型对象

        # 寻找是否有训练过的记录，如果没有则为初次训练，需要从初始点云中初始化每个点对应的三维高斯
        # 以及将每张图片对应的tilt参数dump到 cameras.json 文件中
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "gaussians"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        
        self.train_tilts = {}   # 用于训练的投影序列
        self.test_tilts = {}    # 用于测试的投影序列

        # 根据数据集类型加载场景信息，读取每张图片以及对应的倾斜角度，存储在scene_info变量中
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            print('args.source_path:', args.source_path, '| images:', args.images, '| eval:',  args.eval)
            scene_info = sceneLoadTypeCallbacks["relion"](args.source_path, args.images, args.eval, data)
        elif os.path.exists(os.path.join(args.source_path, "fbp")):
            scene_info = sceneLoadTypeCallbacks["fbp"](args.source_path, args.images, args.eval, data)
        else:
            assert False, "Cound not recognize scene type!"
        
        # 保存点云数据和将每张图片对应的旋转参数到tilt.json文件中
        if not self.loaded_iter:
            # with open(scene_info.mrc_path,'rb') as src_file, open(os.path.join(self.model_path, "input.mrc"), 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            json_tilts = []
            tiltlist = []
            if scene_info.test_tilts:
                tiltlist.extend(scene_info.test_tilts)
            if scene_info.train_tilts:
                tiltlist.extend(scene_info.train_tilts)

            for id, tilt in enumerate(tiltlist):
                json_tilts.append(tilt_to_JSON(id, tilt))

            with open(os.path.join(self.model_path,"tilt.json"),'w') as file:
                json.dump(json_tilts, file)
            
        if shuffle:
            random.shuffle(scene_info.train_tilts)
            random.shuffle(scene_info.test_tilts)

        # 根据resolution_scale加载不同分辨率的训练和测试投影图像
        for resolution_scale in resolution_scales:
            print("Loading Training tilts")
            self.train_tilts[resolution_scale] = tiltList_from_tiltInfos(scene_info.train_tilts, resolution_scale, args)
            print("Loading Testing tilts")
            self.test_tilts[resolution_scale] = tiltList_from_tiltInfos(scene_info.test_tilts, resolution_scale, args)

        # 加载或创建高斯模型
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, 
                                                 "gaussians", 
                                                 "iteration_" + str(self.loaded_iter), 
                                                 "gaussian_points.ply"))
        else: # 根据场景信息中的点云数据创建高斯模型
            print('[debug]: create from fbp')
            self.gaussians.create_from_fbp(scene_info.point_cloud, intensity_threshold=data.intensity_threshold, 
                                           init_size=8000, spatial_lr_scale=1, image_width=data.image_width, image_height=data.image_height, 
                                           grid_size=data.grid_size, volume_x=data.volume_x, volume_y=data.volume_y, volume_z=data.volume_z)

        points = self.gaussians._xyz.detach().cpu().numpy()
        intensities = self.gaussians._intensity.detach().squeeze().cpu().numpy()
        scales = self.gaussians._scaling.detach().cpu().numpy()

        print('3d gaussian have been created...')
        self.gaussians.visualize_gaussians(points, intensities, scales)


    def save(self, iteration):
        """
        保存当前迭代下的3D高斯模型点云
        """
        gaussian_points_path = os.path.join(self.model_path, "gaussians/iteration_{}".format(iteration))
        self.gaussians.save_gaussianpoints(os.path.join(gaussian_points_path, "gaussian_points.ply"))
        self.gaussians.save_gaussianpoints_rgb(os.path.join(gaussian_points_path, "gaussian_points_rgb.ply"))
        self.gaussians.save_csv(os.path.join(gaussian_points_path, "gaussian_points.csv"))
        
        points = self.gaussians._xyz.detach().cpu().numpy()
        intensities = self.gaussians._intensity.detach().cpu().numpy()
        scales = self.gaussians._scaling.detach().cpu().numpy()
        self.gaussians.visualize_gaussians(points, intensities, scales)


    def getTrainTilts(self, scale=1.0): 
        """获取相应分辨率的训练倾斜图像列表"""
        return self.train_tilts[scale]
    
    
    def getTestTilts(self, scale=1.0):
        return self.test_tilts[scale]