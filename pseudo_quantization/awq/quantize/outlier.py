import torch
import torch.nn as nn
import numpy as np
import math

def handle_outlier(w, q_group_size, quant_grid_set, mode_list, outlier_normal_ratio, outlier_config):
    ic, oc = w.shape[1], w.shape[0]
    outlier_mask = torch.zeros_like(w, dtype=torch.bool).to(w.device)
    non_victim_mask = torch.ones_like(w, dtype=torch.bool)

    outlier_type = outlier_config['method']
    outlier_ratio = outlier_config['keep_ratio']
    keep_outlier_num = outlier_config['keep_num']

    def select_identifier(quant_grid_set, mode_list):
        for mode in mode_list:
            if mode == 'pot' or 'flint' in mode or mode == 'float':
                # remove -0 即可
                identifier = None
            else:
                identifier = min(quant_grid_set[mode])
            if identifier is not None:
                mask = (quant_grid_set[mode] != identifier)
                quant_grid_set[mode] = quant_grid_set[mode][mask]
        return quant_grid_set
    
    # 全精度保留 outlier：直接将 outlier value 置 0，再执行量化过程，反量化时加上 FP16 的 outlier
    if "magnitude_weight" in outlier_type:
        outlier_num = math.ceil(ic * outlier_ratio)
        _, outlier_index = torch.topk(w.abs(), outlier_num)
        outlier_mask.scatter_(1, outlier_index, 1)
        outlier_mask = outlier_mask.reshape(w.shape)
        w = w * ~outlier_mask
    # 使用 olive 的方式保留 outlier，outlier 和 victim 都不参与量化；反量化时加上 FP16 的 outlier，victim 为0
    elif outlier_type == "olive":
        num_to_keep = math.ceil(ic * outlier_ratio)
        _, indices = torch.topk(w.abs(), num_to_keep)
        outlier_mask.scatter_(1, indices, 1)

        # 认为 outlier ratio > 某个阈值，才是 outlier
        use_outlier_ratio = True
        if use_outlier_ratio:
            # assert q_group_size != -1
            zero_mask = torch.zeros_like(outlier_normal_ratio, dtype=torch.bool)
            one_mask = torch.ones_like(outlier_normal_ratio, dtype=torch.bool)
            outlier_ratio_mask = torch.where(outlier_normal_ratio > 1.5, one_mask, zero_mask)
            if q_group_size > 0:
                outlier_mask = outlier_mask.reshape(-1, q_group_size)
            outlier_mask = outlier_mask * outlier_ratio_mask
            if q_group_size > 0:
                outlier_mask = outlier_mask.reshape(org_w_shape)

        # olive 需要一个 identifier，quant grid 的size -1
        quant_grid_set = select_identifier(quant_grid_set, mode_list)

        # 使用 olive 的方法得到 victim 的位置
        victim_odd = torch.roll(outlier_mask.view(-1), 1, -1)
        victim_odd[::2] = 0
        victim_even = torch.roll(outlier_mask.view(-1) & (~victim_odd), -1, -1)
        victim_even[1::2] = 0

        non_victim_mask = ~(victim_even | victim_odd)
        non_victim_mask = non_victim_mask.reshape(w.shape)

        # 将 victim value 和 outlier value 置 0，再执行量化过程
        w = w * non_victim_mask
        w = w * ~outlier_mask

    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
        outlier_mask = outlier_mask.reshape(-1, q_group_size)
        non_victim_mask = non_victim_mask.reshape(-1, q_group_size)
    assert w.dim() == 2


    # outlier value 不参与量化，保留 N 个 outlier，用 2 ** n_bit - N 个值进行非均匀量化
    if "group_outlier" in outlier_type:
        
        assert keep_outlier_num == 1 or keep_outlier_num == 0, "only support at most 1 outlier in ANT LUT now"
        # 每个 group 保留 N 个 outlier
        _, indices = torch.topk(w.abs(), keep_outlier_num)
        outlier_mask.scatter_(1, indices, 1)

        # 需要一个 identifier，quant grid 的size -1
        quant_grid_set = select_identifier(quant_grid_set, mode_list)

        w = w * ~outlier_mask
        
    # 使用 olive 的方式保留 outlier，outlier 和 victim 都不参与量化；反量化时加上 FP16 的 outlier，victim 为0
    elif outlier_type == "olive_group":

        # 得到 outlier index
        _, indices = torch.topk(w.abs(), keep_outlier_num)
        outlier_mask.scatter_(1, indices, 1)

        # olive 需要一个 identifier，quant grid 的size -1
        quant_grid_set = select_identifier(quant_grid_set, mode_list)

        # 使用 olive 的方法得到 victim 的位置
        victim_odd = torch.roll(outlier_mask.view(-1), 1, -1)
        victim_odd[::2] = 0
        victim_even = torch.roll(outlier_mask.view(-1) & (~victim_odd), -1, -1)
        victim_even[1::2] = 0

        non_victim_mask = ~(victim_even | victim_odd)
        non_victim_mask = non_victim_mask.reshape(w.shape)
  
        # 将 victim value 和 outlier value 置 0，再执行量化过程
        w = w * non_victim_mask
        w = w * ~outlier_mask

    w = w.reshape(org_w_shape)
    return w, outlier_mask, non_victim_mask, quant_grid_set

def handle_outlier_kmeans(w, outlier_config, normal_float):
    outlier_mask = torch.zeros_like(w, dtype=torch.bool)
    _, indices = torch.topk(w.abs(), outlier_config['keep_num'])
    outlier_mask.scatter_(1, indices, 1)

    # 如果使用 normal + outlier table，需要一个 entry 作为 identifier，聚类中心的数量 -1
    # 如果使用一张 LUT，聚类中心的数量为 2 ** n_bit - 保留 outlier 的数量
    # assert num_cluster >= keep_outlier_num, "num_cluster should be larger than keep_outlier_num"
    identifier = max(normal_float)
    if identifier is not None:
        mask = (normal_float != identifier)
        normal_float = normal_float[mask]

    w = w * ~outlier_mask

    return w, outlier_mask, normal_float