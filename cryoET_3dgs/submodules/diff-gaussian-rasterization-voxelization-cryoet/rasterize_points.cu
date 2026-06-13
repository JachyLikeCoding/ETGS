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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

// 创建并返回一个lambda表达式，该表达式用于调整 torch::Tensor 对象的大小，并返回一个指向它数据的原始指针
std::function<char*(size_t N)> 
resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& intensities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float kernel_size,
	const torch::Tensor& subpixel_offset,
    const int image_height,
    const int image_width,
	const int volume_x,
	const int volume_y,
	const int volume_z,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_density = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts); //(1, H, W) 在指定的视角下，对所有3D高斯进行投影和渲染得到的图像
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32)); // (P,)
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device)); // (0,) 存储所有3D Gaussian对应的参数（均值、尺度、旋转矩阵和密度）的tensor，会动态分配存储空间
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device)); // (0,)
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));  // (0,) 存储在指定视角下渲染得到的图像的tensor，会动态分配存储空间
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);	// 动态调整 geomBuffer 大小的函数，并返回对应的数据指针
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer); //动态调整 binningBuffer 大小的函数，并返回对应的数据指针
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);	// 动态调整 imgBuffer 大小的函数，并返回对应的数据指针
  
  int rendered = 0;
  if(P != 0){
	rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, volume_x, volume_y, volume_z,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		intensities.contiguous().data<float>(),  // 每个高斯的密度
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		kernel_size,
		subpixel_offset.contiguous().data<float>(),
		prefiltered, 							// 是否预先过滤掉了中心点（均值XYZ）不在视野内的3D gaussian，False
		out_density.contiguous().data<float>(),
		radii.contiguous().data<int>(), 		// 存储每个2D gaussian在图像上的半径
		debug);									// False
  }
  return std::make_tuple(rendered, out_density, radii, geomBuffer, binningBuffer, imgBuffer);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& intensities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float kernel_size,
	const torch::Tensor& subpixel_offset,
	const torch::Tensor& dL_dout_density,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_density.size(1);
  const int W = dL_dout_density.size(2);

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dintensities = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dmu = torch::zeros({P, 1}, means3D.options()); 
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(
	  P, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  intensities.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  kernel_size,
	  subpixel_offset.contiguous().data<float>(),
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_density.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dintensities.contiguous().data<float>(),
	  dL_dmu.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dintensities, dL_dmu, dL_dmeans3D, dL_dcov3D, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}