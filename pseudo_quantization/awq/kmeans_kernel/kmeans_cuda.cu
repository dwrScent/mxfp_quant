#include <torch/extension.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <random>
#include <ctime>
#include <cfloat>


__device__ float distance(float a, float b) {
    return fabs(a - b);
}

__global__ void kmeans_kernel(float *input, float *centroids, int *labels, float *output, float *initial_centroids, int N, int K, float* old_centroids, int* data_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    old_centroids = old_centroids + tid * K;
    data_counts = data_counts + tid * K;

    // printf("Thread id\n", tid);
    // 每个线程处理一行数据
    float *data = input + tid * N;
    float *group_centroids;
    if (initial_centroids != nullptr) {
        // 使用自己定义的质心
        group_centroids = initial_centroids + tid * K;  // 为每一行分配K个质心
    }
    else{
        // 使用随机质心
        group_centroids = centroids + tid * K;  // 为每一行分配K个质心
    }
    // 最多迭代的次数，避免过拟合或者无法收敛的情况
    const int max_iterations = 600;
    // float old_centroids[K];
    // int data_counts[K];

    // 初始化
    for(int k = 0; k < K; k++) {
        old_centroids[k] = 0.0f;
    }

    for(int iter = 0; iter < max_iterations; iter++) {
        // 根据当前的质心为每个数据点分配标签
        for(int i = 0; i < N; i++) {
            // 为每个节点找到最近的质心
            float min_dist = FLT_MAX;
            int min_index = -1;
            for(int k = 0; k < K; k++) {
                float dist = distance(data[i], group_centroids[k]);
                if(dist < min_dist) {
                    min_dist = dist;
                    min_index = k;
                }
            }
            labels[tid * N + i] = min_index;
        }

        // 更新质心
        for(int k = 0; k < K; k++) {
            old_centroids[k] = group_centroids[k];
            group_centroids[k] = 0.0f;
            data_counts[k] = 0;
        }

        for(int i = 0; i < N; i++) {
            group_centroids[labels[tid * N + i]] += data[i];
            data_counts[labels[tid * N + i]]++;
        }

        for(int k = 0; k < K; k++) {
            if(data_counts[k] != 0) {
                group_centroids[k] /= data_counts[k];
            }
        }

        // 检查收敛性：如果质心变化很小，就退出
        float centroid_shift = 0.0f;
        for(int k = 0; k < K; k++) {
            centroid_shift += distance(old_centroids[k], group_centroids[k]);
        }

        if(centroid_shift < 1e-5f) {
            break;
        }
    }

    // 输出聚类后的数据
    for(int i = 0; i < N; i++) {
        output[tid * N + i] = group_centroids[labels[tid * N + i]];
    }
    __syncthreads();
}

// PyTorch wrapper function
torch::Tensor kmeans_cuda_forward(torch::Tensor input, torch::Tensor centroids, torch::Tensor labels, torch::Tensor output, torch::Tensor initial_centroids, int group_num, int nums_per_group, int num_cluster) {
    // Extract data pointers from PyTorch tensors

    auto input_ptr = reinterpret_cast<float*>(input.data_ptr<float>());
    auto centroids_ptr = reinterpret_cast<float*>(centroids.data_ptr<float>());
    auto labels_ptr = reinterpret_cast<int*>(labels.data_ptr<int>());
    auto initial_centroids_ptr = reinterpret_cast<float*>(initial_centroids.data_ptr<float>());
    auto output_ptr = reinterpret_cast<float*>(output.data_ptr<float>());

    // std::cout << "before kernel" <<input.device() << centroids.device() << labels.device() << output.device() << initial_centroids.device() << std::endl;

    // 每个 thread 都需要，作为中间变量
    int* data_counts;
    float* old_centroids;
    cudaMallocManaged(&old_centroids, group_num * num_cluster * sizeof(float));
    cudaMallocManaged(&data_counts, group_num * num_cluster * sizeof(int));

    // Calculate grid and block dimensions based on your requirements
    int block_num = group_num > 128 ? group_num / 128 : 1;

    dim3 grid_dim(block_num, 1); // Adjust the grid dimensions as needed
    dim3 block_dim(128, 1); // Adjust the block dimensions as needed

    // printf("kmeans kernel launch\n");

    int device_id = input.device().index();
    // printf("device id: %d\n", device_id);
    cudaSetDevice(device_id);
    // Launch the CUDA kernel
    kmeans_kernel<<<grid_dim, block_dim>>>(input_ptr, centroids_ptr, labels_ptr, output_ptr, initial_centroids_ptr, nums_per_group, num_cluster, old_centroids, data_counts);
    cudaDeviceSynchronize();  // 等待 CUDA 核心完成

    // printf("kmeans kernel done\n");

    cudaFree(old_centroids);
    cudaFree(data_counts);
    // Return the output tensor

    // std::cout << "after kernel" <<input.device() << centroids.device() << labels.device() << output.device() << initial_centroids.device() << std::endl;
    return output;
}
