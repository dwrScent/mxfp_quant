import torch
import torch.nn as nn
from .ant_quant import ant_quantization_search, meta_flint_set, normal_float_value
from .outlier import handle_outlier_kmeans
import kmeans_parallel
# import kmeans_output_mse
# from .group_kmeans import k_means
import time

def random_initializer(input_data, n_clusters):
    batch_size, N = input_data.shape
    centroids = torch.empty(batch_size, n_clusters, device=input_data.device, dtype=input_data.dtype)

    for batch_idx in range(batch_size):
        # 随机打乱输入数据的索引
        shuffled_indices = torch.randperm(N, device=input_data.device)
        
        # 从打乱的索引中选择前K个作为初始质心
        centroids[batch_idx] = input_data[batch_idx, shuffled_indices[:n_clusters]]

    return centroids

@torch.no_grad()
def use_kmeans_quantization(w, w_bit, zero_point=False, q_group_size=-1, outlier_config=None, x_feature=None, max_iter=600, get_labels=False):
    device = w.device
    org_w_shape = w.shape

    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    
    w = w.float()
    # 根据 bit 数决定聚类的 cluster 数
    num_cluster = 2 ** w_bit
    org_w = w.clone().to(device)

    # 根据 bit 数生成 normal float
    # TODO: 除了 4-bit，其他的数值不一定正确
    normal_float = normal_float_value(w_bit)
    # print(normal_float)

    # 每个 group 保留 N 个 outlier 值
    if outlier_config is not None and "group_outlier" in outlier_config['method']:
        w, outlier_mask, normal_float = handle_outlier_kmeans(w, outlier_config, normal_float)
        num_cluster -= 1


    # 设置 kmeans CUDA kernel 函数的参数
    group_num = w.shape[0]
    params_per_group = w.shape[1]
    labels = torch.zeros_like(w, dtype=torch.int32).to(device)
    output = torch.zeros_like(w).to(device)
    centroids = torch.zeros(group_num, num_cluster).to(device)
    
    # 使用 normal_float 作为初始的聚类中心
    assert num_cluster == normal_float.shape[0]
    initial_centroids = normal_float.repeat(group_num, 1).to(device)

    # normalize 到 (-1, 1)
    initial_centroids = torch.where(initial_centroids > 0, initial_centroids / initial_centroids.max(), initial_centroids / (-initial_centroids.min()))
    # 初始聚类中心的值需要 scale 到 weight value 的范围内
    centroids_scale_pos, _ = w.max(dim=1, keepdim=True)
    centroids_scale_neg, _ = w.min(dim=1, keepdim=True)
    initial_centroids = torch.where(initial_centroids > 0, initial_centroids * centroids_scale_pos, initial_centroids * (-centroids_scale_neg))

    # 调用 kmeans CUDA kernel 函数进行聚类
    if x_feature is not None:
        x_feature = x_feature.float().to(device)
        w = kmeans_parallel.weighted_kmeans_cuda(w, x_feature, centroids, labels, output, initial_centroids, group_num, params_per_group, num_cluster, max_iter)
    else:
        w = kmeans_parallel.kmeans_cuda_forward(w, centroids, labels, output, initial_centroids, group_num, params_per_group, num_cluster)

    if outlier_config is not None and "group_outlier" in outlier_config['method']:
        w = w * ~outlier_mask + org_w * outlier_mask

    w = w.reshape(org_w_shape).half()

    # del org_w, labels, output, centroids, initial_centroids

    if get_labels:
        initial_centroids = initial_centroids.half()
        return w, labels, initial_centroids
    else:
        return w

# 分别用 ANT 和 kmeans 进行量化，每个 group 选择量化误差更小的数据类型/查找表
@torch.no_grad()
def ant_kmeans_quant(w_data, w_bit, q_config, ant_config, outlier_config, mse_stats, overall_stats, i, n, get_labels=False):
    mse = nn.MSELoss(reduction='none')
    org_w_shape = w_data.shape
    w_init = w_data.clone()
  
    if 'kmeans' in ant_config['ant_mode']:
        if get_labels:
            w_kmeans_data, kmeans_labels, kmeans_codebook = use_kmeans_quantization(w_data, w_bit, **q_config, outlier_config=outlier_config, get_labels=get_labels)
        else:
            w_kmeans_data = use_kmeans_quantization(w_data, w_bit, **q_config, outlier_config=outlier_config)
        if ant_config["ant_mode"] == "kmeans":
            return (w_kmeans_data, kmeans_labels, kmeans_codebook) if get_labels else w_kmeans_data
    if get_labels:
        w_ant_data, ant_labels, ant_codebook = ant_quantization_search(w_data, w_bit, q_config, ant_config, outlier_config, overall_stats=overall_stats, get_labels=get_labels)
    else:
        w_ant_data = ant_quantization_search(w_data, w_bit, q_config, ant_config, outlier_config, overall_stats=overall_stats)

    if not 'kmeans' in ant_config['ant_mode']:
        ant_codebook = ant_codebook.half()
        return (w_data, ant_labels, ant_codebook) if get_labels else w_data

    q_group_size = q_config['q_group_size']
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w_init = w_init.reshape(-1, q_group_size)
        w_kmeans_data = w_kmeans_data.reshape(-1, q_group_size)
        w_ant_data = w_ant_data.reshape(-1, q_group_size)

    # 根据 kmeans 和 ant 的 MSE 来决定使用哪个
    kmeans_mse = mse(w_init, w_kmeans_data).mean(dim=1, keepdim=True).to(w_init.device)
    ant_mse = mse(w_init, w_ant_data).mean(dim=1, keepdim=True).to(w_init.device)
    kmeans_mask = torch.where(kmeans_mse < ant_mse, torch.ones_like(kmeans_mse, dtype=torch.int8), torch.zeros_like(kmeans_mse, dtype=torch.int8))
    w_data = kmeans_mask * w_kmeans_data + (1 - kmeans_mask) * w_ant_data

    update_statistics(mse_stats, kmeans_mse, ant_mse, kmeans_mask, w_init, w_data, org_w_shape)

    if get_labels:

        total_labels = kmeans_mask * kmeans_labels + (1 - kmeans_mask) * ant_labels
        total_codebook = kmeans_mask * kmeans_codebook + (1 - kmeans_mask) * ant_codebook

    w_data = w_data.reshape(org_w_shape)
    return (w_data, total_labels, total_codebook) if get_labels else w_data
    
def update_mse_stats(w_init, w_quantized, mse_stats, mode):
    mse = nn.MSELoss(reduction='mean')
    mse_value = mse(w_init, w_quantized).cpu()
    mse_stats[mode] += mse_value

def update_statistics(mse_stats, kmeans_mse, ant_mse, choice_mask, w_init, w_data, org_w_shape):
    group_num = w_data.view(-1, org_w_shape[-1]).shape[0]
    kmeans_num, ant_num = choice_mask.sum(), (1 - choice_mask).sum()

    print(f"kmeans mse: {kmeans_mse.mean().item():.9f}, ant mse: {ant_mse.mean().item():.9f}, overall mse: {nn.MSELoss(reduction='mean')(w_init, w_data).item():.9f}")
    print(f"group num: {group_num}, kmeans num: {kmeans_num.item()}, ant num: {ant_num.item()}")

    mse_stats['kmeans'] += kmeans_mse.mean().item()
    mse_stats['ant'] += ant_mse.mean().item()
    mse_stats['overall'] += nn.MSELoss(reduction='mean')(w_init, w_data).item()
    mse_stats['ant_num'] += ant_num.item() / 1e5
    mse_stats['kmeans_num'] += kmeans_num.item() / 1e5