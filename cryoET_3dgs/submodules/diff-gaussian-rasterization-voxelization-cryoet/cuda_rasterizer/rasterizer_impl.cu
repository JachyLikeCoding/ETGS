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
// 将tile+高斯想象成一个组合，按照tile的数量，先排列。如果是同一个tile，按照每个高斯的顺序排序。
#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB on the CPU.
// 寻找给定无符号整数 n 的最高有效位（Most Significant Bit, MSB）的下一个最高位
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(
	int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	// present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
	present[idx] = in_frustum(idx, orig_points, viewmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
// 识别每个tile在排序后的高斯ID列表中的范围，目的是确定哪些高斯ID属于哪个tile，并记录每个tile的开始和结束位置
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}


// 在给定的内存块中初始化 GeometryState 结构
// chunk（一个指向内存块的指针引用），P（元素的数量）
// 使用 obtain 函数为 GeometryState 的不同成员分配空间，并返回一个初始化的 GeometryState 实例
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_intensity, P, 128);
	obtain(chunk, geom.mus, P, 128);
	obtain(chunk, geom.out_density, P, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_density, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}


// 初始化 BinningState 实例，分配所需的内存big进行排序操作
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	// 在GPU上进行基数排序
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P,
	const int X,
	const int Y,
	const int Z,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* intensities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float kernel_size,
	const float* subpixel_offset,
	const bool prefiltered,
	float* out_density,
	int* radii,
	bool debug)
{

	size_t chunk_size = required<GeometryState>(P); // 计算存储所有3D gaussian的各个参数所需要的空间大小
	char* chunkptr = geometryBuffer(chunk_size);    // 给所有3D gaussian的各个参数分配存储空间，并返回存储空间的指针
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);    // 在给定的内存块中初始化 GeometryState结构体，为不同成员分配空间，并返回一个初始化的实例

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;   // 指向radii数据的指针
	}

    // 定义了一个三维网格（dim3是CUDA中定义三维网格维度的数据类型），确定了在水平和垂直方向上需要多少个块来覆盖整个渲染区域
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    // 确定了每个块在X和Y方向上的线程数
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);   // 计算存储所有2D pixel的各个参数所需要的空间大小
	char* img_chunkptr = imageBuffer(img_chunk_size);               // 给所有2D pixel的各个参数分配存储空间，并返回存储空间的指针
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);  //在给定的内存块中初始化 ImageState结构体，为不同成员分配空间，并返回一个初始化的实例

	// std::cout << "Start forward preprocess---" << std::endl;
	// Run preprocessing per-Gaussian (transformation, bounding)
	CHECK_CUDA(FORWARD::preprocess(
		P, X, Y, Z,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		intensities,
		cov3D_precomp,
		viewmatrix, 
        projmatrix,
		width, height,
		kernel_size,
		radii,
		geomState.means2D,  // 存储每个2D gaussian的均值
		geomState.depths,   //存储每个2D gaussian的深度
		geomState.cov3D,  
		out_density,  
		geomState.conic_intensity,    // 每个2D Gaussian的协方差矩阵的逆矩阵
		geomState.mus,
		tile_grid,                  // 在水平和垂直方向上需要多少个块来覆盖整个渲染区域
		geomState.tiles_touched,    // 存储每个2D gaussian覆盖了多少个tile
		prefiltered                 // 是否预先过滤掉了中心点不在视野里的3D gaussian
	), debug)


	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	// 存储所有的2D gaussians总共覆盖了多少个tile
	int num_rendered; 
    // 将 geomState.point_offsets 数组中最后一个元素的值复制到主机内存中的变量 num_rendered
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
    // 将每个3D gaussian对应的 tile index 和深度存到 point_list_keys_unsorted 中
    // 将每个3D gaussian对应的index（第几个gaussian）存储到 point_list_unsorted 中
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
    // 对一个键值对列表进行排序。这里的键值对由 binningState.point_list_keys_unsorted 和 binningState.point_list_unsorted 组成
    // 排序后的结果存储在  binningState.point_list_keys 和 binningState.point_list 中
    // binningState.list_sorting_space 和 binningState.sorting_size 指定了排序操作所需的临时存储空间和其大小。
    // num_rendered 是要排序的元素总数。
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

    // 将 imgState.ranges 数组中的所有元素设为0
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
    // 识别每个 tile 在排序后的高斯ID列表中的范围
    // 目标是确定哪些高斯ID属于哪个tile，并记录每个tile的开始和结束位置
	// std::cout << "num_rendered = " << num_rendered << std::endl;
	// std::cout << "binningState.point_list_keys:" << std::endl;
	// for (int i = 0; i < num_rendered; ++i) {
	// 	std::cout << "point_list_keys[" << i << "] = " << binningState.point_list_keys[i] << std::endl;
	// }

	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)


	// std::cout << "Start FORWARD::render...." << std::endl;

	// Let each tile blend its range of Gaussians independently in parallel
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,    // 排序后的3D gaussian的id列表
		width, height,
		(float2*)subpixel_offset,
		geomState.means2D,          // 每个2D Gaussian在图像上的中心点位置
		intensities,
		geomState.conic_intensity,    // 每个2D Gaussian的协方差矩阵的逆矩阵以及不透明度
		geomState.mus,
		imgState.n_contrib,     // 每个pixel的最后一个贡献的2D Gaussian是谁
		background,             
		out_density), debug)    // 输出图像

	return num_rendered;
}



// Produce necessary gradients for optimization, corresponding to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* intensities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float kernel_size,
	const float* subpixel_offset,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dintensity,
	float* dL_dmu,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);
	// std::cout << "width height: " << width << ", " << height << std::endl;

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// std::cout << "------- Start BACKWARD::render" << std::endl;
	// std::cout << "\n------- tile_grid.x tile_grid.y = "<< tile_grid.x  << ", " << tile_grid.y << std::endl;
	
	// std::cout << "imgState.ranges:" << std::endl;
	// for (int i = 0; i < (tile_grid.x * tile_grid.y); ++i) {
	// 	std::cout << "range[" << i << "] = (" << imgState.ranges[i].x << ", " << imgState.ranges[i].y << ")" << std::endl;
	// }

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	CHECK_CUDA(BACKWARD::render(
		tile_grid,block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		(float2*)subpixel_offset,
		background,
		geomState.means2D,
		geomState.conic_intensity,    // 每个2D Gaussian的协方差矩阵的逆矩阵
		geomState.mus,
		intensities,
		geomState.out_density,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dintensity,
		dL_dmu), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// std::cout << "------- Start BACKWARD::preprocess" << std::endl;
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(
		P,
		(float3*)means3D,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		width, height,
		viewmatrix,
		projmatrix,
		kernel_size,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dmu,
		(glm::vec3*)dL_dmean3D,
		dL_dcov3D,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		geomState.conic_intensity,
	    dL_dintensity), debug)
}