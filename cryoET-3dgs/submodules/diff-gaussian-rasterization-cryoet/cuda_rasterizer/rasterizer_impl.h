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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>


namespace CudaRasterizer
{
    // 用于从内存块中获取特定类型的数据指针，并更新内存块的指针位置。四个参数：
    //     chunk:指向内存块的指针，通过引用传递，表示内存块的起始位置。
    //     ptr:通过引用传递，表示要获取的特定类型T的数据指针的目标地址。
    //     count:表示要获取的数据块的数量
    //     alignment:表示数据的对齐要求，通常为字节对齐的值
    
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
        // 计算偏移量，使得指针满足对齐要求
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
        // 将偏移后的指针转换成T类型指针并赋值给ptr
		ptr = reinterpret_cast<T*>(offset);
        // 更新chunk指针，指向下一个数据块的起始位置
		chunk = reinterpret_cast<char*>(ptr + count);
	}

    // 存储所有3D gaussian的各个参数的结构体
	struct GeometryState
	{
        // 扫描尺寸大小
		size_t scan_size;
        // 指向深度值数组的指针
		float* depths;
        // 指向扫描空间的指针，可能用于存储扫描过程中的临时数据
		char* scanning_space;
        // 指向整数数组的指针，用于存储内部半径信息
		int* internal_radii;
        // 指向 float2 结构体数组的指针，可能用于存储二维均值信息
		float2* means2D;
        // 指向浮点数数组的指针，用于存储三维协方差信息
		float* cov3D;
		float4* conic_intensity;
		float* out_density;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_density;
		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;	// 存储用于排序操作的缓冲区大小
		uint64_t* point_list_keys_unsorted;	// 未排序的键列表
		uint64_t* point_list_keys;			// 排序后的键列表
		uint32_t* point_list_unsorted;		// 未排序的点列表
		uint32_t* point_list;				// 排序后的点列表
		char* list_sorting_space;			// 用于排序操作的缓冲区

		static BinningState fromChunk(char*& chunk, size_t P);
	};


	template<typename T> 
	// 计算存储T类型数据所需的内存大小的函数
	// 通过调用 T::fromChunk 并传递一个空指针来模拟内存分配过程
	// 通过这个过程，它确定了实际所需的内存大小，加上额外的128字节以满足可能的内存对齐要求。
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};