#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define BLOCK_X 16 //定义tile的大小
#define BLOCK_Y 16 
#define NUM_CHANNELS 1 // 原来是3，表示RGB, 现在只有单通道的密度值
#endif

// 定义CUDA渲染器的配置参数，通过这些宏定义，可以方便地在CUDA渲染器的代码中使用这些参数，从而灵活地调整渲染器的输出通道数和线程块大小.