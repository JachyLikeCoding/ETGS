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


// 前向渲染。主要包含两个部分: 
// 1.对每个gaussian进行预处理 得到它在当前视角下的半径、覆盖了哪些tile 
// 2.对每一个tile使用splatting技术渲染得到密度图像


// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, const float* cov3D, const float* viewmatrix)
{
	
    // Parameters:
    //      mean: 三维高斯分布的均值，即高斯分布的中心点
    //      cov3D: 三维协方差矩阵，用于计算二维协方差矩阵
    //      viewmatrix: 视图矩阵，用于将sample坐标系转换为世界坐标系
	
	// 将当前3D gaussian的中心点从sample坐标系投影到世界坐标系
	float3 t = transformPoint4x3(mean, viewmatrix);

	glm::mat3 J = glm::mat3(
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f);

	// 计算视图矩阵的3x3旋转部分
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

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;

    // 返回计算得到的二维协方差矩阵
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}


__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);	// 初始化了一个3x3单位阵
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	// glm::vec4 q = rot;// / glm::length(rot);
	glm::vec4 q = glm::normalize(rot);
    float r = q.w;
    float x = q.x;
    float y = q.y;
    float z = q.z;

    // Compute rotation matrix from quaternion
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );


	glm::mat3 M = R * S;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}



// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
    int P, //高斯分布的点的数量 
	const float* orig_points, 	//三维坐标
	const glm::vec3* scales,  	//缩放
	const float scale_modifier, //缩放调整因子
	const glm::vec4* rotations, //旋转四元数数组
	const float* intensities, 	//每个高斯的密度数组
	const float* cov3D_precomp, //预计算的每个高斯的三维协方差矩阵数组
	const float* viewmatrix, 	//视图矩阵
	const float* projmatrix, 	//投影矩阵
	const int W, int H, 		//输出图像的宽度和高度
	int* radii, 				//每个高斯输出的半径数组
	float2* points_xy_image, 	//每个高斯在图像上的二维坐标
	float* depths, 				//输出的深度
	float* cov3Ds, 				//输出的三维协方差
	float3* conic,
	const dim3 grid, 			//CUDA网格的大小，二维线程块数量
	uint32_t* tiles_touched, 	// 记录每个高斯覆盖的图像块数量的数组
	bool prefiltered) 			//是否对输入进行了预过滤的布尔值
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	radii[idx] = 0;
	tiles_touched[idx] = 0;

	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float3 volume_center = make_float3(512.0f, 512.0f, 50.0f); // 设置正确的中心
	p_orig = p_orig - volume_center;

	// 2. 将体积范围缩放到 [-1, 1]
	float3 min_bound = make_float3(0.0f, 0.0f, 0.0f);  // 根据实际数据设置
	float3 max_bound = make_float3(1024.0f, 1024.0f, 100.0f);     // 根据实际数据设置

	p_orig = (p_orig - min_bound) / (max_bound - min_bound) * 2.0f;

	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float3 p_proj = { p_hom.x, p_hom.y, p_hom.z };
	float2 point_image = { ndc2PixX(p_proj.x, W), ndc2PixY(p_proj.y, H)};
	
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

	float3 cov = computeCov2D(p_orig, cov3D, viewmatrix);

    // 对协方差矩阵进行求逆操作，用于EWA 算法
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f){
		printf("\n\tdet == 0, return");
		return;
	}

	float det_inv = 1.f / det;
	float3 conic_val = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

    // 计算2D协方差矩阵的特征值，用于计算屏幕空间的范围，以确定与之相交的tile。
	float mid = 0.5f * (cov.x + cov.z);
	// 计算长轴的半径
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	// 计算高斯的半径
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    // 有了圆心和半径，16*16瓦片，计算圆覆盖的像素数
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);

	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0){
		return;
	}

	// 存储得到的深度、半径、屏幕坐标等结果，用于下一步继续处理。
	depths[idx] = p_proj.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	conic[idx] = conic_val;
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


template <uint32_t CHANNELS> 
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) 
renderCUDA(
	const uint2* __restrict__ ranges, 			//包含了每个范围的起始和结束索引的数组
	const uint32_t* __restrict__ point_list, 	//包含了点的索引的数组
	int W, int H, 								//图像的宽度和高度
	const float2* __restrict__ points_xy_image, //包含每个点在屏幕上的坐标的数组
	const float* __restrict__ intensities, 		//包含每个点的强度值的数组
	const float3* __restrict__ conic,
	uint32_t* __restrict__ n_contrib, 			//用于存储每个像素的贡献计数的数组
	const float* __restrict__ bg_color, 		//如果提供了背景颜色，将其作为背景
	float* __restrict__ out_density) 			//存储最终渲染结果的数组
{
    // 1. 确定当前像素范围：
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

    // 2. 判断当前线程是否在有效像素范围内：
	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

    // 3.加载点云数据处理范围
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

    // 4. 初始化共享内存
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float3 collected_conic[BLOCK_SIZE];

    // 5. 初始化渲染相关变量：
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float density = 0.0f;

    // 6. 迭代处理点云数据：
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)  // 使用rounds控制循环的迭代次数，每次迭代处理一批点云数据。
	{
        // 检查是否所有线程块都已经完成渲染：统计已经完成渲染的线程数，如果整个线程块都已完成，则跳出循环。
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;
        
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic[block.thread_rank()] = conic[coll_id];
		}
		block.sync(); 

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) 
		{
			contributor++;
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y }; 
			float3 conic = collected_conic[j];
			float power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y; 
			if (power > 0.0f){
				printf("Skipping point at pix_id: %d, power: %f\n", pix_id, power);
				continue;
			}
				
			float sigma_x = rsqrtf(conic.x); // 1/sigma_x
			float sigma_y = rsqrtf(conic.z); // 1/sigma_y
			float rho = conic.y * sigma_x * sigma_y; // pho=con.y / (1/sigma_x * 1/sigma_y)
			float normalization_factor = 1.0f / (2.0f * M_PI * sigma_x * sigma_y * sqrtf(1.0f - rho * rho));

			// 计算点的密度并累加
			int coll_id = collected_id[j];
			float point_density = normalization_factor * expf(power) * intensities[coll_id];
			density += point_density;
			last_contributor = contributor;
		}
	}

    // 7. 写入最终渲染结果：
	if (inside)
	{
		n_contrib[pix_id] = last_contributor;
		out_density[pix_id] = (density + bg_color[0]);
	}
}


// 执行CUDA核函数renderCUDA, 并将结果写入到 n_contrib 和 out_density 数组中。
void FORWARD::render(
	const dim3 grid, dim3 block, 	// 表示CUDA的网格维度、线程块的维度
	const uint2* ranges, 			// 存储了每个线程块需要处理的像素范围
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* intensities,
	const float3* conic,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_density)
{
	// std::cout << "Start renderCUDA---" << std::endl;

	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		intensities,
		conic,
		n_contrib,
		bg_color,
		out_density);
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
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* density, //输出的密度值
	float3* conic,
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
		radii,
		means2D,
		depths,
		cov3Ds,
		conic,
		grid,
		tiles_touched,
		prefiltered
		);
}
