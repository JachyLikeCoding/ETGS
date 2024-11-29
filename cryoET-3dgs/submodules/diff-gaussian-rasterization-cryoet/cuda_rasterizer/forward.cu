#include "forward.h"
#include <iostream>
#include <fstream>
#include <iostream>
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

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



// 前向渲染。主要包含两个部分: 
// 1.对每个gaussian进行预处理 得到它在当前视角下的半径、覆盖了哪些tile 
// 2.对每一个tile使用splatting技术渲染得到密度图像

// Forward version of 2D covariance matrix computation
__device__ float4 computeCov2D(const float3& mean,  float kernel_size, const float* cov3D, const float* viewmatrix)
{
	
    // Parameters:
    //      mean: 三维高斯分布的均值，即高斯分布的中心点
    //      cov3D: 三维协方差矩阵，用于计算二维协方差矩阵
    //      viewmatrix: 视图矩阵，用于将sample坐标系转换为世界坐标系

	glm::mat3 J = glm::mat3(
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f);

	// 计算视图矩阵的3x3旋转部分，用于线性变换
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]
	);

	// 对于平行投影，投影变换是线性的，直接使用视图矩阵的旋转部分
	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]
	);

	glm::mat3 cov = glm::transpose(T) * Vrk * T;
	
	// compute the coef of alpha based on the detemintant
	const float det_0 = max(1e-6, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
	const float det_1 = max(1e-6, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);
	float coef = sqrt(det_0 / (det_1+1e-6) + 1e-6);

	if (det_0 <= 1e-6 || det_1 <= 1e-6){
		coef = 0.0f;
	}
	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += kernel_size;
	cov[1][1] += kernel_size;

    // 返回计算得到的二维协方差矩阵
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]), float(coef) };
}


// 这段代码用于将每个高斯函数的尺度和旋转属性转换成世界空间中的三维协方差矩阵，并确保四元数的归一化
// 参数说明：
//      scale：表示高斯函数的缩放属性。包含了高斯函数在三个轴上的缩放比例
//      mod： 表示高斯函数的修正因子。用于调整缩放属性，通常用于控制高斯函数的尺度
//      rot: 表示高斯函数的旋转属性。这个四元数描述了高斯函数在旋转过程中的姿态
//      cov3D: 指向输出三维协方差矩阵的指针。这个矩阵用于描述高斯函数在世界空间中的分布情况
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
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
	cov3D[0] = Sigma[0][0]; // Cov(xx)
	cov3D[1] = Sigma[0][1]; // Cov(xy)
	cov3D[2] = Sigma[0][2]; // Cov(xz)
	cov3D[3] = Sigma[1][1]; // Cov(yy)
	cov3D[4] = Sigma[1][2]; // Cov(yz)
	cov3D[5] = Sigma[2][2]; // Cov(zz)
}



// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
    int P, 						//高斯分布的点的数量 
	const float* orig_points, 	//三维坐标
	const glm::vec3* scales,  	//缩放因子数组
	const float scale_modifier, //缩放因子修正值
	const glm::vec4* rotations, //旋转四元数数组
	const float* intensities, 	//每个高斯的密度数组
	const float* cov3D_precomp, //预计算的3D协方差矩阵
	const float* viewmatrix, 	//视图矩阵
	const float* projmatrix, 	//投影矩阵
	const int W, int H, 		//输出图像的宽度和高度
	const float kernel_size,
	int* radii, 				//每个高斯输出的半径数组
	float2* points_xy_image, 	//每个高斯在图像上的二维坐标
	float* depths, 				//输出的深度
	float* cov3Ds, 				//输出的三维协方差
	float* density, 			//输出的密度值
	float4* conic_intensity,	//圆锥参数和密度数组			
	const dim3 grid, 			//CUDA网格的大小，二维线程块数量
	uint32_t* tiles_touched, 	//记录每个高斯覆盖的图像块数量的数组
	bool prefiltered) 			//是否对输入进行了预过滤
{
	auto idx = cg::this_grid().thread_rank(); // 对应的线程
	if (idx >= P)
		return;

    // 首先初始化了一些变量，用于跟踪半径和被覆盖的tile数量。初始化为0，如果不被改变那么这个高斯将不会进一步处理。
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// 对原始点进行投影变换到视图空间
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	// float3 p_view;
	// // Perform near culling, quit if outside. 自定义平行投影裁剪条件
	// if (!in_frustum(idx, orig_points, viewmatrix, false, p_view))
	// 	return;

	// 2. 将体积范围缩放到 [-1, 1]
	// TODO: 假设体积的最大范围是 [min_bound, max_bound]，需要提前计算
	float3 min_bound = make_float3(0.0f, 0.0f, 0.0f);  // 根据实际数据设置
	// float3 max_bound = make_float3(1024.0f, 1024.0f, 100.0f);     // 根据实际数据设置(cryoet.yaml)
	float3 max_bound = make_float3(1024.0f, 1024.0f, 512.0f);     // 根据实际数据设置(10164.yaml)

	// 缩放到 [-1, 1]
	p_orig = (p_orig - min_bound) / (max_bound - min_bound) * 2.0f;

    // 将世界坐标转换到假设xy范围一致的投影坐标 (NDC)，
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	
	// 平行投影中无需透视除法
	float3 p_proj = { p_hom.x, p_hom.y, p_hom.z };

    // 根据输入的缩放和旋转参数，计算3D协方差矩阵。
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

    // 基于平行投影计算2D屏幕空间的协方差矩阵。（计算投影到二维后椭圆的样子）
	float4 cov = computeCov2D(p_orig, kernel_size, cov3D, viewmatrix);

    // 对协方差矩阵进行求逆操作，用于EWA 算法
	float det = (cov.x * cov.z - cov.y * cov.y); // 行列式
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

    // 计算2D协方差矩阵的特征值，用于计算屏幕空间的范围，以确定与之相交的tile。
	float mid = 0.5f * (cov.x + cov.z);
	// 计算长轴的半径
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	// 计算高斯的半径
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    // 从NDC坐标系转到像素坐标系
	float2 point_image = { ndc2PixX(p_proj.x, W), ndc2PixY(p_proj.y, H)}; //p_proj.x p_proj.y的范围应该是[-1,1]，像素坐标系就是图像大小范围

    // 有了圆心和半径，16*16瓦片，计算圆覆盖的像素数
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid); // 获取一个矩形
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0){ // 说明这个椭球不存在
		return;
	}

	// 存储得到的深度、半径、屏幕坐标等结果，用于下一步继续处理。
	depths[idx] = p_proj.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	conic_intensity[idx] = { conic.x, conic.y, conic.z, intensities[idx] * cov.w}; // 二维协方差矩阵的逆矩阵
	// conic_intensity[idx] = { conic.x, conic.y, conic.z, intensities[idx]}; // 二维协方差矩阵的逆矩阵

	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

	// 添加调试信息
    if (idx == 8000) {
		printf("\n[forward] orig_points = %f, %f, %f ", orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
		printf("\n[forward] scale = %f, %f, %f, scale_modifier= %f", scales[idx].x, scales[idx].y, scales[idx].z, scale_modifier);
		printf("\n[forward] rotation = %f, %f, %f, %f", rotations[idx].x, rotations[idx].y, rotations[idx].z, rotations[idx].w);
        printf("\n[forward] cov3D = %f, %f, %f, %f, %f, %f", cov3D[0], cov3D[1], cov3D[2], cov3D[3], cov3D[4], cov3D[5]);
        printf("\n[forward] cov2D = %f, %f, %f", cov.x, cov.y, cov.z);
		printf("\n[forward] intensity = %f", intensities[idx]);
		printf("\n[forward] conic = %f, %f, %f", conic.x, conic.y, conic.z);
		printf("\n[forward] lambda1 = %f, lambda2 = %f", lambda1, lambda2);
		printf("\n[forward] my_radius = %f, point_image = (%f, %f)", my_radius, point_image.x, point_image.y);
		printf("\n[forward] (rect_max.x = %d, rect_min.x = %d, rect_max.y = %d, rect_min.y = %d)", rect_max.x, rect_min.x, rect_max.y, rect_min.y);
		printf("\n[forward] depths = %f, radii = %d, tiles_touched = %d\n", depths[idx], radii[idx], tiles_touched[idx]);
	}
}



// 声明了一个CUDA核函数renderCUDA,用于执行主要的光栅化过程。每个线程块协作处理一个图块，每个线程处理一个像素。在数据获取和光栅化之间交替：
template <uint32_t CHANNELS> // 模板参数CHANNELS 代表输出的通道数
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) 
// 使用CUDA启动限制，这是CUDA启动核函数时使用的线程格和线程块的数量。
renderCUDA(
	const uint2* __restrict__ ranges, 			//点范围数组
	const uint32_t* __restrict__ point_list, 	//点索引数组
	int W, int H, 								//图像宽度和高度
	const float2* __restrict__ subpixel_offset,
	const float2* __restrict__ points_xy_image, //点在图像上的坐标数组
	const float* __restrict__ intensities, 		//每个点的强度值的数组
	const float4* __restrict__ conic_intensity, //圆锥参数和密度数组		
	uint32_t* __restrict__ n_contrib, 			//每个像素的贡献计数的数组
	const float* __restrict__ bg_color, 		//背景颜色数组
	float* __restrict__ out_intensity) 			//最终渲染结果的数组
{
	// Identify current tile and associated min/max pixel range.
    // 1. 确定当前像素范围：
    //   确定当前线程块要处理的像素范围，包括pix_min pix_max，并计算当前线程对应的像素坐标 pix
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // 向上取整操作
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

    // 2. 判断当前线程是否在有效像素范围内：
    //   根据像素坐标判断当前线程是否在有效的图像范围内，如果不在，则将done设置为true，表示该线程不需要执行渲染操作。
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;
	// add the offset to pixel
	if (inside){
		pixf.x += subpixel_offset[pix_id].x;
		pixf.y += subpixel_offset[pix_id].y;
		// if (pix_id == 0){
		// 	printf("\n\n in forward rendering, pixf is %.5f %.5f  offset %.5f %.5f\n\n", pixf.x, pixf.y, subpixel_offset[pix_id].x, subpixel_offset[pix_id].y);
		// }
	}

    // 3.加载点云数据处理范围
    // 这部分代码加载当前线程块要处理的点云数据的范围，即ranges数组中对应的范围，并计算点云数据的迭代批次rounds和总共要处理的点数 todo
	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

    // 4. 初始化共享内存
    // 分别定义三个共享内存数组，用于在每个线程块内共享数据。
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_intensity[BLOCK_SIZE];

    // 5. 初始化渲染相关变量：
    // 包括当前密度、贡献者数量等。
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float intensity_accum = 0.0f;

    // 6. 迭代处理点云数据：
    // 在每个迭代中处理一批点云数据。内部循环迭代每个点进行渲染计算，并更新密度信息。
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)  // 使用rounds控制循环的迭代次数，每次迭代处理一批点云数据。
	{
        // 检查是否所有线程块都已经完成渲染：统计已经完成渲染的线程数，如果整个线程块都已完成，则跳出循环。
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;
        
        // 共享内存中获取点云数据：
        // 每个线程通过索引progress计算要加载的点云数据的索引 coll_id，然后从全局内存中加载到共享内存 collected_id, collected_xy,collected_conic 中。
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_intensity[block.thread_rank()] = conic_intensity[coll_id];
		}
		block.sync(); // 确保所有线程都加载完成

        // 迭代处理当前批次的点云数据
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) //在当前批次的循环中，每个线程处理一条点云数据。
		{
			contributor++;
            // 计算当前点在屏幕上的坐标xy与当前像素坐标pixf的差值，并使用锥体参数计算power
			// Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y }; // 当前点和像素中心的差值
			float4 conic_intensity = collected_conic_intensity[j]; // 用来存储二维高斯函数的参数
			float power = -0.5f * (conic_intensity.x * d.x * d.x + conic_intensity.z * d.y * d.y) - conic_intensity.y * d.x * d.y; // 计算高斯函数的指数部分

			if (power > 0.0f)
				continue;
			
			// 在cryoET场景中，将高斯函数的指数计算结果作为密度贡献
			// float intensity_contrib = min(0.99f, intensities[collected_id[j]] * exp(power)); // 这里计算的是当前点的密度（不能直接算成高斯中心的值）
			float intensity_contrib = min(0.99f, conic_intensity.w * exp(power)); // 这里计算的是当前点的密度（不能直接算成高斯中心的值）
			if (intensity_contrib < 0.00001f)
				continue;

			intensity_accum += intensity_contrib;

			// Keep track of last range entry to update this pixel.
			last_contributor = contributor;
		}
	}

    // 7. 写入最终渲染结果：
    //  如果当前线程在有效像素范围内，则将最终的渲染结果写入相应的缓冲区，包括 n_contrib 和 out_intensity
	if (inside)
	{
		n_contrib[pix_id] = last_contributor;
		out_intensity[pix_id] = max(intensity_accum, 0.00001f);
		if(pix_id == 8000){
			printf("\n intensity_accum: %f", intensity_accum);
			printf("\n Writing to out_intensity [before]: density: %f", out_intensity[pix_id]);
		}	
		// out_intensity[pix_id] = fminf(fmaxf(out_intensity[pix_id], 0.0f), 1.0f);
		// if(pix_id == 8000){
		// 	printf("\nWriting to out_intensity [after]: density: %f\n", out_intensity[pix_id]);
		// }	
	}
}


// 执行CUDA核函数renderCUDA, 并将结果写入到 n_contrib 和 out_intensity 数组中。
void FORWARD::render(
	const dim3 grid, dim3 block, 	// 表示CUDA的网格维度、线程块的维度
	const uint2* ranges, 			// 存储了每个线程块需要处理的像素范围
	const uint32_t* point_list,
	int W, int H,
	const float2* subpixel_offset,
	const float2* means2D,
	const float* intensities,
	const float4* conic_intensity,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_intensity)
{
	// std::cout << "Start renderCUDA---" << std::endl;
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		subpixel_offset,
		means2D,
		intensities,
		conic_intensity,
		n_contrib,
		bg_color,
		out_intensity);
}

void FORWARD::preprocess(
	int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* intensities,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float kernel_size,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* density, //输出的密度值
	float4* conic_intensity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	// std::cout << "Start preprocessCUDA---" << std::endl;

	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
		intensities,
		cov3D_precomp,
		viewmatrix, 
		projmatrix,
		W, H,
		kernel_size,
		radii,
		means2D,
		depths,
		cov3Ds,
		density,
		conic_intensity,
		grid,
		tiles_touched,
		prefiltered
		);
}
