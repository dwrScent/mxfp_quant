#include <torch/extension.h>

torch::Tensor kmeans_cuda_forward(torch::Tensor input, torch::Tensor centroids, torch::Tensor labels, torch::Tensor output, torch::Tensor initial_centroids, int group_num, int nums_per_group, int num_cluster) ;
