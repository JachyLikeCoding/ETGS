#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
    // Perform initial steps for each gaussian prior to rasterization.
    // 光栅化之前对每个高斯做一些初始化的步骤
    void preprocess(
        int P,
        const float* orig_points,
        const glm::vec3* scales,
        const float scale_modifier, //缩放调整因子
        const glm::vec4* rotations, //旋转四元数数组
        const float* intensities,
        const float* cov3D_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const int W, int H,
        const float kernel_size,
        int* radii,
        float2* points_xy_image,
        float* depths,
        float* cov3Ds,
        float* density, //输出的密度值
        float4* conic_intensity,
        const dim3 grid,
        uint32_t* tiles_touched,
        bool prefiltered);
    
    // Main rasterization method
    void render(
        const dim3 grid, dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        const float2* subpixel_offset,
        const float2* points_xy_image,
        const float* intensities,
        const float4* conic_intensity,
        uint32_t* n_contrib,
        const float* bg_color,
        float* out_density);
}

#endif