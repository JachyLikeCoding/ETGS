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

 // 用于存放辅助功能或辅助类型的声明、定义或实现。

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

__forceinline__ __device__ float ndc2PixX(float v, float W)
{
    // 将NDC X坐标从 [-1, 1] 转换到像素坐标范围 [0, W]
	return ((v + 1.0) * W - 1.0) * 0.5;
}

__forceinline__ __device__ float ndc2PixY(float u, float H)
{
    // 将NDC Y坐标从 [-1, 1] 转换到像素坐标范围 [0, H]
    return ((u + 1.0) * H - 1.0) * 0.5;
}


// 将设备坐标转换为像素坐标
__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}


// 计算当前的2D GAUSSIAN落在哪几个tile上
__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}


__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum0(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

__forceinline__ __device__ bool in_frustum(int idx,
    const float* orig_points,
    const float* viewmatrix,
    bool prefiltered,
    float3& p_view)
{
    // 读取原始点的坐标
    float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

    // 将点转换到相机空间（平行投影）
    p_view = transformPoint4x3(p_orig, viewmatrix);


    // 平行投影不需要深度范围检查，仅需检查点在投影平面上的坐标是否在合理范围内
    // 检查范围：[-1, 1]（归一化设备坐标，NDC）
    if (p_view.x < -256 || p_view.x > 256 || p_view.y < -256 || p_view.y > 256) 
    {
        if (prefiltered)
        {
            printf("Point is filtered although prefiltered is set. This shouldn't happen!");
            __trap();
        }
        return false;
    }

    return true;
}



// 加法
__device__ inline float3 operator+(const float3 &a, const float3 &b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// 减法
__device__ inline float3 operator-(const float3 &a, const float3 &b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
// 定义 float3 与 float 之间的减法操作
__device__ inline float3 operator-(const float3 &a, const float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

// 乘法
__device__ inline float3 operator*(const float3 &a, const float3 &b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

// 标量乘法
__device__ inline float3 operator*(const float3 &a, const float &b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

// 标量除法
__device__ inline float3 operator/(const float3 &a, const float &b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}
// 按分量除法
__device__ inline float3 operator/(const float3 &a, const float3 &b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}



// CHECK_CUDA 这个宏的作用是简化在CUDA编程中对CUDA函数调用的返回值进行错误检查的过程。通过使用这个宏，可以在调试模式下方便地进行CUDA错误检查和处理，
// 提高代码的可维护性和可读性。注意，在生产环境中可能需要关闭调试模式以避免不必要的性能损失。宏的实现逻辑如下：
// 1. 执行参数A 表示的CUDA函数调用或者表达式
// 2. 如果启用了调试模式debug，则执行以下步骤：
// 	调用 cudaDeviceSynchronize() 同步设备上的所有流，确保之前的CUDA操作已经完成。
// 	检查同步的返回值 ret，如果不是cudaSuccess， 表示出现了CUDA错误
// 	打印出错信息，包括出错文件、行号以及CUDA错误信息
// 	抛出 std::runtime_error 异常，并传递 CUDA 错误信息作为异常消息。

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif