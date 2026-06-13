/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */


// 生成三个Python可调用的接口函数：
// rasterize_gaussians，对应 rasterize_points.cu/.h 中的RasterizeGaussiansCUDA
// rasterize_gaussians_backward，对应rasterize_points.cu/.h 中的RasterizeGaussiansBackwardCUDA
// mark_visible 对应rasterize_points.cu/.h 中的markVisible


#include <torch/extension.h>
#include "rasterize_points.h"
#include "voxelize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("voxelize_gaussians", &VoxelizeGaussiansCUDA);
  m.def("voxelize_gaussians_backward", &VoxelizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}