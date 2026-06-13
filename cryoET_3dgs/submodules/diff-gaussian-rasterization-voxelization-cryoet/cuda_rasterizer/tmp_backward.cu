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

#include "backward.h"
#include <fstream>
#include "auxiliary.h"
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// 用于计算逆2D协方差矩阵的反向传播版本。是一个CUDA核函数，用于在反过程中计算损失函数相对于均值和协方差的梯度。
__global__ void computeCov2DCUDA(
    int P, //高斯数量
	const float3* means, //每个高斯的均值(中心位置)
	const int* radii,   //每个高斯点在2D投影中的半径
	const float* cov3Ds,    //每个高斯的3D协方差矩阵
	int width, int height,
	const float* view_matrix,	//视图矩阵，用于将3D点转换到2D屏幕空间
	const float* dL_dconics, // 损失函数相对于逆协方差矩阵的梯度
	float3* dL_dmeans,      // 损失函数相对于高斯点均值的梯度
	float* dL_dcov)         // 损失函数相对于高斯点3D协方差矩阵的梯度
{
	// 每个线程处理一个高斯点，超出范围或不可见的点直接返回
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// 当前高斯点的3D协方差矩阵，每个高斯点的3D协方差矩阵占用6个浮点数
	const float* cov3D = cov3Ds + 6 * idx;

	// 提取梯度并重新计算2D协方差矩阵
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };

	glm::mat3 J = glm::mat3(1.0f, 0.0f, 0.0f,
							0.0f, 1.0f, 0.0f,
							0.0f, 0.0f, 0.0f);
	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;
	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	// 计算协方差矩阵的逆矩阵和梯度
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	if (idx == 0) {
		printf("Thread %d: cov3D = [%f, %f, %f, %f, %f, %f]\n", idx, cov3D[0], cov3D[1], cov3D[2], cov3D[3], cov3D[4], cov3D[5]);
		printf("a: %f, b: %f, c: %f\n", a, b, c);
		printf("cov2D: (%f, %f, %f, %f)\n", cov2D[0][0], cov2D[0][1], cov2D[1][0], cov2D[1][1]);
	}

	float denom = a * c - b * b; //计算行列式
	float eps = 1e-6;
	if (fabs(denom) < eps) {
		denom = (denom < 0 ? -eps : eps); // 保证 denom 不为零
	}
	
	float dL_da = 0, dL_db = 0, dL_dc = 0;

	float denom2inv = 1.0f / ((denom * denom) + eps);
	// 确保 denom2inv 不为零，以避免后续计算的数值不稳定
	if (fabs(denom2inv) < eps) {
		denom2inv = (denom2inv < 0 ? -eps : eps);
	}


	if (denom2inv != 0){
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;

	// Gradients of loss w.r.t. J matrix (only the upper 2x3 part).
	float dL_dJ00 = dL_dT00 * W[0][0] + dL_dT01 * W[1][0];
	float dL_dJ01 = dL_dT00 * W[0][1] + dL_dT01 * W[1][1];
	float dL_dJ10 = dL_dT10 * W[0][0] + dL_dT11 * W[1][0];
	float dL_dJ11 = dL_dT10 * W[0][1] + dL_dT11 * W[1][1];

	// Update dL_dmeans based on gradients w.r.t. J
	float dL_dtx = dL_dJ00;
	float dL_dty = dL_dJ11;
	
	// // Account for transformation of mean to t
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx/P, dL_dty/P, 0.0f }, view_matrix);


    // 损失相对于均值的梯度
	// float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, 0.0f }, view_matrix);

	dL_dmeans[idx] = dL_dmean;

	// 调试信息输出
	if(idx == 1000){
		printf("mean: (%f, %f, %f)\n", mean.x, mean.y, mean.z);
		printf("dL_dconic: (%f, %f, %f)\n", dL_dconic.x, dL_dconic.y, dL_dconic.z);
		printf("cov2D: (%f, %f, %f, %f)\n", cov2D[0][0], cov2D[0][1], cov2D[1][0], cov2D[1][1]);
		printf("denom: %f\n", denom);
		printf("denom2inv: %f\n", denom2inv);
		printf("dL_da: %f, dL_db: %f, dL_dc: %f\n", dL_da, dL_db, dL_dc);
		printf("dL_dmean: (%f, %f, %f)\n", dL_dmean.x, dL_dmean.y, dL_dmean.z);
	}
}



// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(
	int idx, 
	const glm::vec3 scale, 
	float mod, 
	const glm::vec4 rot, 
	const float* dL_dcov3Ds, 
	glm::vec3* dL_dscales, 
	glm::vec4* dL_drots)
{

	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// 使用四元数生成旋转矩阵
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	// 将每个元素的协方差损失梯度转换成矩阵形式
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M 计算损失梯度相对于矩阵M的梯度
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale 计算损失梯度相对于尺度的梯度
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	// 更新 dL_dMt 以考虑尺度变化
	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion 计算损失梯度相对归一化四元数的梯度
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion 计算损失梯度相对于未归一化四元数的梯度
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}




// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P,
	const float3* means,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* view,
	const float* proj,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}




// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float3* __restrict__ conic,
	const float* __restrict__ intensities,
	const float* __restrict__ out_density,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dintensities)
{
	// printf("------- Start renderCUDA, W=%d H=%d\n", W, H);
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;

	// 还有多少3D Gaussian要处理
	int toDo = range.y - range.x;
	// printf("--------- toDo = %d, range.y=%d, range.x=%d\n", toDo, range.y, range.x);
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float3 collected_conic[BLOCK_SIZE];
	__shared__ float collected_intensities[BLOCK_SIZE];

	const float density_final = inside ? out_density[pix_id] : 0;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 2.0f / W;
	const float ddely_dy = 2.0f / H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic[block.thread_rank()] = conic[coll_id];
			collected_intensities[block.thread_rank()] = intensities[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			// 跟踪当前高斯的ID,如果这个高斯位于这个像素的最后一个贡献者之后就跳过。
			contributor--;
			if (contributor >= last_contributor)
				continue;

			const float2 xy = collected_xy[j];
			const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			const float3 conic = collected_conic[j];
			const float power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
			
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float density_contribution = G;

			// 过滤掉一些影响过小的高斯点
			if (density_contribution < 1.0f/255.0f)
				continue;

			const int global_id = collected_id[j];
			const float dL_dpixel = dL_dpixels[pix_id];
			
			// Helpful reusable temporary variables
			// const float dL_dG = conic.w * dL_dpixel;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * conic.x - gdy * conic.y;
			const float dG_ddely = -gdy * conic.z - gdx * conic.y;

			// 更新关于高斯密度值的梯度
			atomicAdd(&dL_dintensities[global_id], dL_dpixel * density_contribution);

			// Update gradients w.r.t. 2D mean position of the Gaussian 更新关于高斯的2D均值位置的梯度
			atomicAdd(&dL_dmean2D[global_id].x, dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric) 更新关于2D协方差的梯度
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y);
			atomicAdd(&dL_dconic2D[global_id].z, -0.5f * gdy * d.y);

		}
	}
}



void BACKWARD::preprocess(
	int P,
	const float3* means3D,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	int W, int H,
	const float* viewmatrix,
	const float* projmatrix,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		W, H,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P,
		(float3*)means3D,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3Ds,
		viewmatrix,
		projmatrix,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcov3D,
		dL_dscale,
		dL_drot);
}



void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float3* conic,
	const float* intensities,
	const float* out_density,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float3* dL_dconic2D,
	float* dL_dintensities)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic,
		intensities,
		out_density,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dintensities
		);
}