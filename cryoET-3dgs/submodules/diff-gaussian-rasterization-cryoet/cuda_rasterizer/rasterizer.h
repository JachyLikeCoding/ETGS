#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>


// 这是一个静态类 Rasterizer, 包含了多个静态成员函数，用于进行渲染器相关的操作
// 这个头文件定义了一个CUDA渲染器的接口和功能，可以在其他文件中包含该头文件并使用 CudaRasterizer::Rasterizer 类的静态成员函数来进行相关的渲染和计算任务。
// 其中，静态成员函数的具体作用如下：
// markVisible:标记可见性，用于判断哪些点在视野内
// forward: 前向渲染，生成图像或几何数据
// backward: 反向传播，计算梯度以进行优化或学习

namespace CudaRasterizer
{
    class Rasterizer
    {
        public:
            static void markVisible(
                int P,
                float* means3D,
                float* viewmatrix,
                float* projmatrix,
                bool* present
            );

            static int forward(
                std::function<char* (size_t)> geometryBuffer,
                std::function<char* (size_t)> binningBuffer,
                std::function<char* (size_t)> imageBuffer,
                const int P, 
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
                int* radii = nullptr,
                bool debug = false);

            static void backward(
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
                char* image_buffer,
                const float* dL_dpix,
                float* dL_dmeans2D,
                float* dL_dconic,
                float* dL_dintensities,
                float* dL_dmeans3D,
                float* dL_dcov3D,
                float* dL_dscales,
                float* dL_drotations,
                bool debug);
    };
};

#endif