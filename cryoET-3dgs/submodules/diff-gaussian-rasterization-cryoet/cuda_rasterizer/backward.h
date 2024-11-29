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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* subpixel_offset,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_intensity,
		const float* intensities,
		const float* out_density,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dintensities);

	void preprocess(
		int P,
		const float3* means,
		const int* radii,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		int W, int H,
		const float* view,
		const float* proj,
		const float kernel_size,
		const float3* dL_dmean2D,
		const float* dL_dconic,
		glm::vec3* dL_dmeans,
		float* dL_dcov3D,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		const float4* conic_intensity,
		float* dL_dintensity);
}

#endif