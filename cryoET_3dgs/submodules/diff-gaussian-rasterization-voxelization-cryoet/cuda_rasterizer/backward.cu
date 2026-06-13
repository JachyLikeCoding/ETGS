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
	const float h_x, float h_y,
	const float kernel_size,
	const float* view_matrix,	//视图矩阵，用于将3D点转换到2D屏幕空间
	const float* dL_dconics, // 损失函数相对于逆协方差矩阵的梯度
	const float* dL_dmus,
	float3* dL_dmeans,      // 输出：损失函数相对于高斯点均值的梯度
	float* dL_dcov,         // 输出：损失函数相对于高斯点3D协方差矩阵的梯度
	const float4* __restrict__ conic_intensity,
	float* dL_dintensity)
{
	// 每个线程处理一个高斯点，超出范围或不可见的点直接返回
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// 读取当前高斯点的3D协方差矩阵，占用6个浮点数
	const float* cov3D = cov3Ds + 6 * idx;

	// 提取梯度，并重新计算2D协方差矩阵
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float dL_dmu = dL_dmus[idx]; 
	const float4 conic = conic_intensity[idx];

	// 使用视图矩阵变换均值
	float3 t = transformPoint4x3(mean, view_matrix);

	// 用于变换的视图矩阵子集
	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);
	// 3D空间中的协方差矩阵
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W;
	// T =  [T00 T01 T02]
	//		[T10 T11 T12]

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;
	// Use helper variables for 2D covariance entries. More compact.
	float hata = cov2D[0][0] += 0.0f;
	float hatb = cov2D[0][1];
	float hatc = cov2D[0][2];
	float hatd = cov2D[1][1] += 0.0f;
	float hate = cov2D[1][2];
	float hatf = cov2D[2][2];

	float dL_da = 0, dL_db = 0, dL_dc = 0, dL_dd = 0, dL_de = 0, dL_df = 0;
	float denom = hata * hatd - hatb * hatb;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
	float diamond = hata * hatd - hatb * hatb;
	
	//mu gradient
	float circ = hata * hatd * hatf + 2 * hatb * hatc * hate - hata * hate * hate - hatf * hatb * hatb - hatd * hatc * hatc;
	float mu_square = 2 * M_PI * circ / diamond;
	float mu = 0.0f;
	if (mu_square > 0.0f){
		mu = sqrt(2 * M_PI * circ / diamond);
	}
	float pi_mu = M_PI / (mu + 0.0000001f);
	float circ_diamond = circ / diamond;

	if (denom2inv != 0.0f  && mu != 0.0f){
		// 计算损失相对于2D协方差矩阵条目的梯度	
		// 二维协方差矩阵的三个参数a,b,c对应的是以下矩阵形式：
		// C =  [a b]
		//		[b c]
		// 损失函数L对应的是通过逆变换(即矩阵的逆)求出一个二次曲线矩阵（conic matrix）, 假设有损失函 L=f(C^-1)
		// 要计算损失函数对协方差矩阵中a,b,c的梯度，首先要计算C的逆矩阵
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-hatd * hatd * dL_dconic.x + 2 * hatb * hatd * dL_dconic.y + (denom - hata * hatd) * dL_dconic.z);
		dL_dc = denom2inv * (-hata * hata * dL_dconic.z + 2 * hata * hatb * dL_dconic.y + (denom - hata * hatd) * dL_dconic.x);
		dL_db = denom2inv * 2 * (hatb * hatd * dL_dconic.x - (denom + 2 * hatb * hatb) * dL_dconic.y + hata * hatb * dL_dconic.z);
		
		dL_da += pi_mu * ((hatd * hatf - hate * hate) / diamond -  hatd * circ_diamond / diamond) * dL_dmu;
		dL_db += pi_mu * ((2 * hatc * hate - 2 * hatf * hatb) / diamond + 2 * hatb * circ_diamond / diamond) * dL_dmu;
		dL_dc += pi_mu * ((2 * hatb * hate - 2 * hatd * hatc) / diamond) * dL_dmu;
		dL_dd += pi_mu * ((hata * hatf - hatc * hatc) / diamond -  hata *circ_diamond / diamond) * dL_dmu;
		dL_de += pi_mu * ((2 * hatb * hatc - 2 * hata * hate) / diamond) * dL_dmu;
		dL_df += pi_mu * ((hata * hatd - hatb * hatb) / diamond) * dL_dmu;

		// if (det_0 <= 1e-6 || det_1 <= 1e-6){
		// 	dL_dintensity[idx] = 0;
		// } else {
		// 	dL_da += dcoef_da;
		// 	dL_dc += dcoef_dc;
		// 	dL_db += dcoef_db;
		// 	dL_dintensity[idx] = dL_dintensity[idx] * coef;
		// }


		// 计算损失相对于3D协方差矩阵条目Vrk的梯度	
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);
		// cov2D = transpose(T) * transpose(Vrk) * T;
		// 非对角元素有双重梯度
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		// 如果分母为0，则将梯度设为0
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}
	
	// Compute gradients w.r.t. the mean using simplified transformation
	// float3 dL_dmean = { h_x * dL_da, h_y * dL_dc, 0.0f };
	// dL_dmeans[idx] = dL_dmean;

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T
	// 梯度计算：二维投影协方差矩阵和损失函数针对中间矩阵T的梯度的关系
	// T传递梯度给J，J传递梯度给mean3D
	// float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
	// 	(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	// float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
	// 	(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	// float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
	// 	(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	// float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
	// 	(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	// float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
	// 	(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	// float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
	// 	(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// // 将相对于变换矩阵和均值的梯度结合
    // float3 dL_dmean = {
    //     view_matrix[0] * dL_dT00 + view_matrix[1] * dL_dT01 + view_matrix[2] * dL_dT02,
    //     view_matrix[4] * dL_dT10 + view_matrix[5] * dL_dT11 + view_matrix[6] * dL_dT12,
    //     view_matrix[8] * (dL_dT00 + dL_dT01 + dL_dT02) + view_matrix[9] * (dL_dT10 + dL_dT11 + dL_dT12)};

    // // 将计算出的梯度分配到输出数组
    // dL_dmeans[idx] = dL_dmean;
	// if (idx == 0)
    // 	printf("dL_dz: %.4f\n", dL_dmeans[0].z); // 应始终为0
	// 调试信息输出
	// if(idx == 8000){
	// 	printf("\n[backward] Thread %d: cov3D = [%f, %f, %f, %f, %f, %f]\n", idx, cov3D[0], cov3D[1], cov3D[2], cov3D[3], cov3D[4], cov3D[5]);
	// 	printf("[backward] cov2D: (%f, %f, %f, %f)\n", cov2D[0][0], cov2D[0][1], cov2D[1][0], cov2D[1][1]);
	// 	printf("[backward] a: %f, b: %f, c: %f\n", a, b, c);
	// 	printf("[backward] mean: (%f, %f, %f)\n", mean.x, mean.y, mean.z);
	// 	printf("[backward] dL_dconic: (%f, %f, %f)\n", dL_dconic.x, dL_dconic.y, dL_dconic.z);
	// 	printf("[backward] denom: %f\n", denom);
	// 	printf("[backward] denom2inv: %f\n", denom2inv);
	// 	printf("[backward] dL_da: %f, dL_db: %f, dL_dc: %f\n", dL_da, dL_db, dL_dc);
	// 	printf("[backward] dL_dmean: (%f, %f, %f)\n", dL_dmean.x, dL_dmean.y, dL_dmean.z);
	// }
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
	// printf("------- Start computeCov3D, idx=%d", idx);
	// Recompute (intermediate) results for the 3D covariance computation.
	// 重新计算用于3D协方差计算的中间结果
	glm::vec4 q = rot;// / glm::length(rot);
	q = glm::normalize(q);
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

	// 构建缩放矩阵S
	glm::mat3 S = glm::mat3(1.0f);
	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	// 计算矩阵M
	glm::mat3 M = S * R; // 3*3 矩阵

	// 获得损失相对于3D协方差矩阵的梯度
	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// 三维协方差矩阵的梯度计算
	// 将每个元素的协方差损失梯度转换成矩阵形式
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0],         0.5f * dL_dcov3D[1],  0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1],  dL_dcov3D[3],         0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2],  0.5f * dL_dcov3D[4],  dL_dcov3D[5]
	);

	// 计算损失梯度相对于矩阵M的梯度
	// dSigma_dM = 2 * M  因为sigma是M去掉了对称部分，也就是sigma是M的对称部分，sigma=M*M^T
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	// 求L对M的导数，为了求下面缩放矩阵S的梯度
	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// 计算损失梯度相对于尺度的梯度
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]); // 矩阵的迹 trace
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	// 更新 dL_dMt 以考虑尺度变化(反向更新)
	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// 计算损失梯度相对归一化四元数的梯度
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// 计算损失梯度相对于未归一化四元数的梯度
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });

	// 调试信息输出
	// if (idx == 8000) {
	// 	printf("\n[backward] dL_dcov3D = [%f, %f, %f, %f, %f, %f]\n", 
	// 		dL_dcov3D[0], dL_dcov3D[1], dL_dcov3D[2], 
	// 		dL_dcov3D[3], dL_dcov3D[4], dL_dcov3D[5]);

	// 	printf("[backward] dL_dscale: (%f, %f, %f)\n", 
	// 		dL_dscale->x, dL_dscale->y, dL_dscale->z);

	// 	printf("[backward] dL_dq: (%f, %f, %f, %f)\n", 
	// 		dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w);
		
	// 	printf("[backward] dL_drot: (%f, %f, %f, %f)\n", 
	// 		dL_drot->x, dL_drot->y, dL_drot->z, dL_drot->w);
	// }

}


// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// 反向传播预处理步骤，除了协方差计算和反演（这些由先前的内核调用处理）
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
	// 获取其在CUDA栅格中的线程索引idx，用于确定要处理的gaussian的索引。
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	// 处理屏幕空间点的梯度
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// 损失函数对三维均值的梯度。首先从三维点经过投影矩阵得到二维点的过程开始，然后利用链式法则计算梯度。
	// 假设一个3D点m=(mx my mz) 通过投影矩阵P(4*4矩阵)投影到2D空间。包含视图和投影变换。
	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w; // mul1 = (P00 mx + P01 my + P02 mz + P03) mw^2
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation of cov2D affects it.
	// 这是均值梯度的第二部分，之前的cov2D计算也会影响它
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
	const float2* __restrict__ subpixel_offset,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_intensity,
	const float* __restrict__ mus,
	const float* __restrict__ intensities,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dintensities,
	float* __restrict__ dL_dmu)
{

	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;
	
	// if (inside){
	// 	pixf.x += subpixel_offset[pix_id].x;
	// 	pixf.y += subpixel_offset[pix_id].y;
	// }
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_intensity[BLOCK_SIZE];
	__shared__ float collected_intensities[BLOCK_SIZE];
	__shared__ float collected_mu[BLOCK_SIZE];
	
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float dL_dpixel = inside ? dL_dpixels[pix_id] : 0;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport coordinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

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
			collected_conic_intensity[block.thread_rank()] = conic_intensity[coll_id];
			// collected_intensities[block.thread_rank()] = intensities[coll_id];
			collected_mu[block.thread_rank()] = mus[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// 跟踪当前高斯的ID,如果这个高斯位于这个像素的最后一个贡献者之后就跳过。
			contributor--;
			if (contributor >= last_contributor)
				continue;

			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 conic_intensity = collected_conic_intensity[j];
			float mu = collected_mu[j];
			const float power = -0.5f * (conic_intensity.x * d.x * d.x + conic_intensity.z * d.y * d.y) - conic_intensity.y * d.x * d.y;

			if (power > 0.0f){
				// printf("[backward] Skipping point at pix_id: %d, power: %f\n", pix_id, power);
				continue;
			}

			const float G = exp(power); // 高斯分布的衰减指数
			const float alpha = conic_intensity.w * mu * G;
			if (alpha <0.00001f)
				continue;
			
			const int global_id = collected_id[j];
			float dL_dcontrib = 1.0f * dL_dpixel;

			// 更新关于高斯的2D均值位置和2D协方差的梯度
			const float dL_dG = conic_intensity.w * mu * dL_dcontrib;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * conic_intensity.x - gdy * conic_intensity.y;
			const float dG_ddely = -gdy * conic_intensity.z - gdx * conic_intensity.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// we use this new metric for densification, please check https://arxiv.org/pdf/2404.10772.pdf Densification section for more details.
			// const float abs_dL_dmean2D = abs(dL_dG * dG_ddelx * ddelx_dx) + abs(dL_dG * dG_ddely * ddely_dy);
			// atomicAdd(&dL_dmean2D[global_id].z, abs_dL_dmean2D);

			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);
			
			// 更新关于高斯密度值的梯度
			atomicAdd(&dL_dintensities[global_id], mu * G * dL_dcontrib);
			atomicAdd(&(dL_dmu[global_id]), conic_intensity.w * G * dL_dcontrib);
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
	const float kernel_size,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_dmu,
	glm::vec3* dL_dmean3D,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	const float4* conic_intensity,
	float* dL_dintensity)
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
		kernel_size,
		viewmatrix,
		dL_dconic,
		dL_dmu,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		conic_intensity,
		dL_dintensity);

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
	const float2* subpixel_offset,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_intensity,
	const float* mus,
	const float* intensities,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dintensities,
	float* dL_dmu)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		subpixel_offset,
		bg_color,
		means2D,
		conic_intensity,
		mus,
		intensities,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dintensities,
		dL_dmu
		);
}