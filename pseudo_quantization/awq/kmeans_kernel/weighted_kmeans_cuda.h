#include <torch/extension.h>

torch::Tensor weighted_kmeans_cuda(torch::Tensor input, torch::Tensor x_feature, torch::Tensor centroids, torch::Tensor labels, torch::Tensor output, torch::Tensor initial_centroids, int group_num, int nums_per_group, int num_cluster, int max_iter) ;
