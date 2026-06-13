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
#include <stdio.h> // for debug
#include "forward.h"
#include <iostream>
#include <fstream>
#include <iostream>
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


__device__ static float atomicMax(float* address, float val)
{
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do{
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
		__float_as_int(::fmaxf(val, __int_as_float(assumed))));

	}while (assumed != old);
	return __int_as_float(old);
}


// 基于高斯点的体素化前向计算过程，主要用于将3D高斯分布的点云数据转换成体素表示。

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
// 计算3D高斯分布的协方差矩阵
static __device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
    q = glm::normalize(q);

	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];  // a
	cov3D[1] = Sigma[0][1];  // b
	cov3D[2] = Sigma[0][2];  // c
	cov3D[3] = Sigma[1][1];  // d
	cov3D[4] = Sigma[1][2];  // e
	cov3D[5] = Sigma[2][2];  // f
}


// 预处理每个高斯点，将其从世界坐标系转换到体素坐标系，并计算相关的协方差矩阵和半径
template<int C>
__global__ void preprocessCUDA(
    int P,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* intensities,
	const float* cov3D_precomp,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z, //体素网格的尺寸
	const float sVoxel_x, float sVoxel_y, float sVoxel_z, //体素网格的物理尺寸
	const float center_x, float center_y, float center_z, //体素网格的中心坐标
	int* radii_x, int* radii_y, int* radii_z, // 每个高斯点在体素空间中的半径
	float3* points_xyz_vol, //转换后的体素空间坐标
	float* depths,			//每个高斯点的深度
	float* cov3Ds,			//存储每个高斯点的协方差矩阵
	float* conic_intensity,	//存储每个高斯点的圆锥强度
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
    auto idx = cg::this_grid().thread_rank(); // idx
	if (idx >= P) return;

	radii_x[idx] = 0;
	radii_y[idx] = 0;
	radii_z[idx] = 0;
	tiles_touched[idx] = 0;
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// 根据体素的总尺寸和分辨率计算每个体素在实际空间中的尺寸
	float dVoxel_x = sVoxel_x;
	float dVoxel_y = sVoxel_y;
	float dVoxel_z = sVoxel_z;

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}else{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Transfer to voxel space 将协方差矩阵从原始空间转换到体素空间
	// 通过体素的分辨率进行尺度转换得到新的协方差矩阵，它将适应体素网格的分辨率。
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 M = glm::mat3(
		1.f/sVoxel_x, 0.0f, 0.0f,
		0.0f, 1.f/sVoxel_y, 0.0f,
		0.0f, 0.0f, 1.f/sVoxel_z);

	glm::mat3 cov = glm::transpose(M) * (Vrk) * M;

	float hata = cov[0][0];
	float hatb = cov[0][1];
	float hatc = cov[0][2];
	float hatd = cov[1][1];
	float hate = cov[1][2];
	float hatf = cov[2][2];
	float det = hata * hatd * hatf + 2 * hatb * hatc * hate - hata * hate * hate - hatf * hatb * hatb - hatd * hatc * hatc;
	if (fabsf(det) < 1e-8f)
    	return;


	float det_inv = 1.f / det;
	
	float inv_a = (hatd * hatf - hate * hate) * det_inv;
	float inv_b = (hatc * hate - hatb * hatf) * det_inv;
	float inv_c = (hatb * hate - hatc * hatd) * det_inv;
	float inv_d = (hata * hatf - hatc * hatc) * det_inv;
	float inv_e = (hatb * hatc - hata * hate) * det_inv;
	float inv_f = (hata * hatd - hatb * hatb) * det_inv;

	glm::vec3 scale = scales[idx];
	// printf("\nscales: %f, %f, %f\n", scale.x, scale.y, scale.z);

	float max_scale = scale_modifier * max(max(scale.x, scale.y), scale.z);
	// float sigma_x = sqrtf(cov[0][0]);
	// float sigma_y = sqrtf(cov[1][1]);
	// float sigma_z = sqrtf(cov[2][2]);
	// sigma_x = max(sigma_x, 1.0f);
	// sigma_y = max(sigma_y, 1.0f);
	// sigma_z = max(sigma_z, 1.0f);
	float sigma_x = fmaxf(sqrtf(fabsf(cov[0][0])), 1.0f);
	float sigma_y = fmaxf(sqrtf(fabsf(cov[1][1])), 1.0f);
	float sigma_z = fmaxf(sqrtf(fabsf(cov[2][2])), 1.0f);

	float3 my_radius = {
		max(ceil(4.f * sigma_x),1.0f),
		max(ceil(4.f * sigma_y),1.0f),
		max(ceil(4.f * sigma_z),1.0f)
	};

	float3 point_vol = {(p_orig.x - center_x) / sVoxel_x + nVoxel_x * 0.5f, 
						(p_orig.y - center_y) / sVoxel_y + nVoxel_y * 0.5f,
						(p_orig.z - center_z) / sVoxel_z + nVoxel_z * 0.5f};


	// if (idx == 100){
	// 	printf("\nscales: %f, %f, %f\n", scale.x, scale.y, scale.z);
	// 	printf("\nOriginal point: %f, %f, %f", p_orig.x, p_orig.y, p_orig.z);
	// 	printf("\nVoxel space: %f, %f, %f", point_vol.x, point_vol.y, point_vol.z);
	// 	printf("\nRadius: %f, %f, %f", my_radius.x, my_radius.y, my_radius.z);
	// 	printf("\nCovariance matrix: %f, %f, %f", cov3D[0], cov3D[3], cov3D[5]);
	// }

    if (point_vol.x + my_radius.x < 0 || point_vol.y + my_radius.y < 0 || point_vol.z + my_radius.z < 0 || point_vol.x - my_radius.x > (float)nVoxel_x || point_vol.y - my_radius.y > (float)nVoxel_y || point_vol.z - my_radius.z > (float)nVoxel_z)
	{
		return;
	}

	uint3 cube_min, cube_max;
	getCube(point_vol, my_radius, cube_min, cube_max, grid);
	// printf("\ngrid: %d, %d, %d", grid.x, grid.y, grid.z);

	// if ((cube_max.x - cube_min.x) * (cube_max.y - cube_min.y) * (cube_max.z - cube_min.z) == 0)
	// 	return;
	// 更新每个点的半径以及影响的体素数量
	radii_x[idx] = my_radius.x;
	radii_y[idx] = my_radius.y;
	radii_z[idx] = my_radius.z;
	tiles_touched[idx] = (cube_max.z - cube_min.z) * (cube_max.y - cube_min.y) * (cube_max.x - cube_min.x);
	
	depths[idx] = p_orig.z;  // just give a value
	points_xyz_vol[idx] = point_vol;
	conic_intensity[idx * 7 + 0] = inv_a;
	conic_intensity[idx * 7 + 1] = inv_b;
	conic_intensity[idx * 7 + 2] = inv_c;
	conic_intensity[idx * 7 + 3] = inv_d;
	conic_intensity[idx * 7 + 4] = inv_e;
	conic_intensity[idx * 7 + 5] = inv_f;
	conic_intensity[idx * 7 + 6] = intensities[idx];
	// printf("\n....finish preprocessCUDA");

}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK3D_X * BLOCK3D_Y * BLOCK3D_Z) // 8*8*8
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float3* __restrict__ points_xyz_vol,
	const float* __restrict__ conic_intensity,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_volume
	)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();

	uint32_t horizontal_blocks1 = (nVoxel_x + BLOCK3D_X - 1) / BLOCK3D_X;
	uint32_t horizontal_blocks2 = (nVoxel_y + BLOCK3D_Y - 1) / BLOCK3D_Y; // 16*16
	
	uint3 voxel_min = { 
		block.group_index().x * BLOCK3D_X, 
		block.group_index().y * BLOCK3D_Y,  
		block.group_index().z * BLOCK3D_Z
	};
	uint3 voxel_max = { 
		min(voxel_min.x + BLOCK3D_X, nVoxel_x), 
		min(voxel_min.y + BLOCK3D_Y , nVoxel_y),  
		min(voxel_min.z + BLOCK3D_Z , nVoxel_z)
	};
	uint3 voxel = { 
		voxel_min.x + block.thread_index().x, 
		voxel_min.y + block.thread_index().y, 
		voxel_min.z + block.thread_index().z
	};
	uint32_t voxel_id = nVoxel_z * nVoxel_y * voxel.x 
						+ nVoxel_z * voxel.y 
						+ voxel.z;

	// add 0.5 because we donnot count offset previously, like gs code in ndc2pixel()
	float3 voxelf = { (float)voxel.x + 0.5f, (float)voxel.y + 0.5f, (float)voxel.z + 0.5f};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = voxel.x < nVoxel_x && voxel.y < nVoxel_y && voxel.z < nVoxel_z;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().z * horizontal_blocks2 * horizontal_blocks1 + block.group_index().y * horizontal_blocks1 + block.group_index().x];
	
	const int rounds = ((range.y - range.x + BLOCK3D_SIZE - 1) / BLOCK3D_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK3D_SIZE];
	__shared__ float3 collected_xyz[BLOCK3D_SIZE];
	__shared__ float collected_conic_a[BLOCK3D_SIZE];
	__shared__ float collected_conic_b[BLOCK3D_SIZE];
	__shared__ float collected_conic_c[BLOCK3D_SIZE];
	__shared__ float collected_conic_d[BLOCK3D_SIZE];
	__shared__ float collected_conic_e[BLOCK3D_SIZE];
	__shared__ float collected_conic_f[BLOCK3D_SIZE];
	__shared__ float collected_inten[BLOCK3D_SIZE];

	// Initialize helper variables
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C = 0;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK3D_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK3D_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK3D_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];

			collected_id[block.thread_rank()] = coll_id;
			collected_xyz[block.thread_rank()] = points_xyz_vol[coll_id];
			collected_conic_a[block.thread_rank()] = conic_intensity[coll_id * 7 + 0];
			collected_conic_b[block.thread_rank()] = conic_intensity[coll_id * 7 + 1];
			collected_conic_c[block.thread_rank()] = conic_intensity[coll_id * 7 + 2];
			collected_conic_d[block.thread_rank()] = conic_intensity[coll_id * 7 + 3];
			collected_conic_e[block.thread_rank()] = conic_intensity[coll_id * 7 + 4];
			collected_conic_f[block.thread_rank()] = conic_intensity[coll_id * 7 + 5];
			collected_inten[block.thread_rank()] = conic_intensity[coll_id * 7 + 6];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK3D_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			float3 xyz = collected_xyz[j];
			float3 d = { xyz.x - voxelf.x, xyz.y - voxelf.y, xyz.z - voxelf.z };
			float conic_a = collected_conic_a[j];
			float conic_b = collected_conic_b[j];
			float conic_c = collected_conic_c[j];
			float conic_d = collected_conic_d[j];
			float conic_e = collected_conic_e[j];
			float conic_f = collected_conic_f[j];
			float inten = collected_inten[j];
			float power = - 0.5f * (conic_a * d.x * d.x + conic_d * d.y * d.y + conic_f * d.z * d.z) 
						- conic_b * d.x * d.y - conic_c * d.x * d.z - conic_e * d.y * d.z;
			
			if (power > 0.0f)
				continue;
			
			float power_clamped = max(power, -80.0f);
			float intensity_contrib = inten * exp(power_clamped);
				
			// float intensity_contrib = inten * exp(power);

			// Simply add all intensities
			C += intensity_contrib;

			// Keep track of last range entry to update this pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		n_contrib[voxel_id] = last_contributor;
		// out_volume[voxel_id] += C;
		atomicAdd(&out_volume[voxel_id], C);

		// out_volume[voxel_id] += 0.05f;  // 比如 0.01
	}
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float3* means3D_norm,
	const float* conic_intensity,
	uint32_t* n_contrib,
	float* out_volume)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		nVoxel_x, nVoxel_y, nVoxel_z,
		means3D_norm,
		conic_intensity,
		n_contrib,
		out_volume
		);
	
}


void FORWARD::preprocess(
	int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* intensities,
	const float* cov3D_precomp,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float sVoxel_x, float sVoxel_y, float sVoxel_z,
	const float center_x, float center_y, float center_z,
	int* radii_x, int* radii_y, int* radii_z,
	float3* means3D_norm,
	float* depths,
	float* cov3Ds,
	float* conic_intensity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered
	)
{	
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
		intensities,
		cov3D_precomp,
		nVoxel_x, nVoxel_y, nVoxel_z,
		sVoxel_x, sVoxel_y, sVoxel_z,
		center_x, center_y, center_z,
		radii_x, radii_y, radii_z,
		means3D_norm,
		depths,
		cov3Ds,
		conic_intensity,
		grid,
		tiles_touched,
		prefiltered
		);
	// printf("\nAFTER kernel launch: P :%d\n", P);
}
