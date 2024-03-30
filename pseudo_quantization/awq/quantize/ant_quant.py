import torch
import torch.nn as nn
import numpy as np
import os,sys
import datetime
from ..utils.make_distribution import  outlier_ratio_stat
from .outlier import handle_outlier

def print_time(print_str):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{timestamp} - {print_str}')

meta_flint_set = {
# 'flint_0': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0],
# 'nf4': [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0],
# 'nf4_3bit': [-1.0, -0.5250730514526367, -0.28444138169288635, -0.18477343022823334, 0.0, 0.16093020141124725, 0.33791524171829224, 0.5626170039176941],
# 'nf4_olive': [-0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]

# 'codebook_int': [-64.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 56.0],
# 'codebook_flint': [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 48.0, -48.0],
# 'codebook_pot': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 32.0, -32.0, 512.0, -512.0],
# 'codebook_flint_0': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 48.0, -48.0],

# 'flint_1': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 8.0, -8.0],
# 'flint_2': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 12.0, -12.0],
# 'flint_3': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0],
# 'flint_4': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 2.5, -2.5, 3.0, -3.0, 3.5, -3.5],
# 'flint_5': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0, 12.0, -12.0],
'flint_6': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0, 16.0, -16.0],
'flint_7': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 24.0, -24.0],
# 'flint_8': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 32.0, -32.0],
# 'flint_9': [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0, 7.0, -7.0],
# 'flint_10': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0, 12.0, -12.0],
'flint_11': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0, 16.0, -16.0],
'flint_12': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 24.0, -24.0],
# 'flint_13': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 32.0, -32.0],
# 'flint_14': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0, 7.0, -7.0],
# 'flint_15': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 12.0, -12.0, 16.0, -16.0, 24.0, -24.0],
# 'flint_16': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 12.0, -12.0, 16.0, -16.0, 32.0, -32.0],
# 'flint_17': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 32.0, -32.0, 48.0, -48.0],
# 'flint_18': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0, 32.0, -32.0, 64.0, -64.0],
# 'flint_19': [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 10.0, -10.0, 12.0, -12.0, 14.0, -14.0],
# 'flint_20': [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0],
# 'flint_21': [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 4.0, -4.0],
# 'flint_22': [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0],
# 'flint_23': [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0],
# 'flint_24': [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 1.25, -1.25, 1.5, -1.5, 1.75, -1.75],
# 'flint_25': [0.0, 0.125, -0.125, 0.25, -0.25, 0.375, -0.375, 0.5, -0.5, 0.625, -0.625, 0.75, -0.75, 0.875, -0.875],
}


def int_value(n_bit, signed=True):
    B = n_bit - 1 if signed else n_bit
    values = [0.] + list(range(1, 2 ** B))
    if signed:
        values += [-i for i in range(1, 2 ** B)]
        values.append(-2 ** B)
    return values

def pot_value(n_bit, signed=True):
    B = n_bit - 1 if signed else n_bit
    exp_bit = B
    values = []
    values.append(0.)
    values.append(-0.)
    for i in range(0, 2 ** exp_bit - 1):
        values.append(2 ** i)
        if signed:
            values.append(-2 ** i)

    return values

def flint_value(n_bit, signed=True, exp_base=0):

    B = n_bit - 1 if signed else n_bit

    value_bit = B
    assert(value_bit >= 2)

    exp_num =     value_bit * 2 - 1
    neg_exp_num = value_bit - 1
    pos_exp_num = value_bit - 1
    
    
    exp_max = pos_exp_num + exp_base
    exp_min = -neg_exp_num

    ## Append zero value
    values = [0., -0.]

    # values = [0.]
    ## exponent negative
    for i in range(0, neg_exp_num + 1):
        exp_bit = i + 2
        exp_value = -(exp_bit - 1)
        mant_bit = value_bit - exp_bit
        for j in range(int(2 ** mant_bit)):
            v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if signed:
                values.append(-v)

    ## exponent zero
    exp_bit = 2
    exp_value = 0
    mant_bit = value_bit - exp_bit
    for j in range(int(2 ** mant_bit)):
        v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
        values.append(v)
        if signed:
            values.append(-v)

    ## exponent positive     
    for i in range(1, pos_exp_num):
        exp_bit = i + 2
        exp_value = i
        mant_bit = value_bit - exp_bit
        for j in range(int(2 ** mant_bit)):
            v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if signed:
                values.append(-v)
    ## Append max value
    values.append(2 ** exp_max)
    if signed:
        values.append(-2 ** exp_max)

    return values

def float_value(n_bit, signed=True, exp_field=2):
    B = n_bit - 1 if signed else n_bit

    # mapping, total_bit: exponent_bit
    exp_field_map = {3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
    if n_bit in exp_field_map:
        exp_field = exp_field_map[n_bit]
    else:
        raise ValueError("Not support this bit width")
    exp_bit = exp_field

    man_bit = B - exp_bit
    values = []
    min_to_zero = True
    subnormal = True
    for i in range(2 ** exp_bit):
        for j in range(2 ** man_bit):
            if min_to_zero:
                values.append(0.)
                values.append(-0.)
                min_to_zero = False
            else:
                if subnormal:
                    values.append((2 ** i) * (j * 2 ** (-man_bit)))
                else:
                    values.append((2 ** (i - 1)) * (1 + j * 2 ** (-man_bit)))

                if signed:
                    if subnormal:
                        values.append(-(2 ** i) * (j * 2 ** (-man_bit)))
                    else:
                        values.append(-(2 ** (i - 1)) * (1 + j * 2 ** (-man_bit)))
        subnormal = False

    return torch.tensor(values)
from scipy.stats import norm
def normal_float_value(n_bit, signed=True, offset=0.9677083, use_extra_value=True):

    if use_extra_value:
        # one more positive value, this is an asymmetric type
        v1 = norm.ppf(torch.linspace(offset, 0.5, 2 ** (n_bit - 1) + 1)[:-1]).tolist()
        v2 = [0] ## we have 15 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 2 ** (n_bit - 1))[:-1])).tolist()
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, 2 ** (n_bit - 1))[:-1]).tolist()
        v2 = [0] ## we have 14 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 2 ** (n_bit - 1))[:-1])).tolist()

    v = v1 + v2 + v3
    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()

    assert values.numel() == 2 ** n_bit 
    # print(values)
    return values

def generate_quant_grid(n_bit=4, signed=True, ant_mode="flint"):
    quant_grid_set = {}
    quant_grid_funcs = {
        "int": int_value,
        "flint": flint_value,
        "pot": pot_value,
        "float": float_value,
        "nf": normal_float_value,
    }
    mode_list = ant_mode.split('-')
    if "kmeans" in mode_list:
        mode_list.remove("kmeans")
    if "weighted_kmeans" in mode_list:
        mode_list.remove("weighted_kmeans")

    for mode in mode_list:
        if mode in quant_grid_funcs:
            quant_grid_set[mode] = quant_grid_funcs[mode](n_bit=n_bit, signed=signed)
        elif mode == 'meta_flint':
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}")

    if 'meta_flint' in ant_mode:
         quant_grid_set.update(meta_flint_set)

    # Convert list to tensor
    for key, value in quant_grid_set.items():
        quant_grid_set[key] = torch.tensor(value)
    return quant_grid_set


@torch.no_grad()
def get_quant_weight(w, quant_grid, mode="int", zero_point=False, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False):
    '''
    return : dequantized weight, mse?
    '''
    quant_grid = quant_grid.to(w.device)

    if zero_point:
        assert 0, "Not support zero point in ant quant now."
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        if mode == "int":
            max_quant_val = max(abs(quant_grid)) * 2 - 1
        else:
            max_quant_val = max(abs(quant_grid)) * 2
        scales = (max_val - min_val).clamp(min=1e-5) / max_quant_val
        # fp16 z.p.
        zeros = (-min_val ) + 1e-5
        quant_grid = quant_grid + max(abs(quant_grid))
    else:
        max_val = w.abs().amax(dim=1, keepdim=True)

        if pos_value is None or pos_value == True:
            max_quant_val = max(quant_grid)
        elif pos_value == False:
            max_quant_val = abs(min(quant_grid))
        else:
            raise NotImplementedError 
        
        # Compute the scaling factor
        scales = (max_val * alpha) / max_quant_val
        # print(scales)
        zeros = 0

    labels = (((w + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # print(labels, labels.shape, quant_grid, quant_grid.shape)
    w_deq = quant_grid[labels] * scales - zeros

    
    quant_mse = (w_deq-w).abs().pow(2)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    if get_labels:
        return w_deq, quant_mse_sum, labels, quant_grid * scales
    else:
        return w_deq, quant_mse_sum


@torch.no_grad()
def ant_quantization(w, n_bit=8, zero_point=False, q_group_size=-1, ant_mode="flint-int-pot", ant_search_granularity=1, w_low=None, w_high=None, ant_asym=None, outlier_config=None, outlier_normal_ratio=None, alpha=1.0, display=True, overall_stats={}, pos_value=None, get_labels=False):
    '''
    Input:weight, n_bit, q_group_size, ant_mode
    Output:Dequantize-quantized weight 
    '''

    org_w_shape = w.shape
    mode_list = ant_mode.split('-')
    
    # kmeans 不在这个部分做
    if "kmeans" in mode_list:
        mode_list.remove("kmeans")

    # Generate quantization grid values for different data types
    quant_grid_set = generate_quant_grid(n_bit, ant_mode=ant_mode)
    if 'meta_flint' in mode_list:
        mode_list.remove('meta_flint')
        mode_list.extend(meta_flint_set.keys())

    # 对 outlier 进行处理
    org_w = w.clone()
    if outlier_config['method'] != "none":
        w, outlier_mask, non_victim_mask, quant_grid_set = handle_outlier(w, q_group_size, quant_grid_set, mode_list, outlier_normal_ratio, outlier_config)

    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
        org_w = org_w.reshape(-1, q_group_size)
        # outlier_group_mask = outlier_group_mask.reshape(-1, q_group_size)
    assert w.dim() == 2
    
    w_deq_list = {}
    quant_mse_list = {}
    labels_list = {}
    codebook_list = {}
    exist_mode = ""
    # 获取所有 data type 的 w_deq 以及其量化 MSE
    for mode in mode_list:
        if get_labels:
            w_deq_list[mode], quant_mse_list[mode], labels_list[mode], codebook_list[mode] = get_quant_weight(w, quant_grid_set[mode], mode=mode, zero_point=zero_point, q_group_size=q_group_size, alpha=alpha, pos_value=pos_value, get_labels=get_labels)
        else:
            w_deq_list[mode], quant_mse_list[mode] = get_quant_weight(w, quant_grid_set[mode], mode=mode, zero_point=zero_point, q_group_size=q_group_size, alpha=alpha, pos_value=pos_value)
        exist_mode = mode
    
    # 用于记录每个 group 选择 data type 的信息
    data_type_identify = torch.zeros_like(quant_mse_list[exist_mode], dtype=torch.int32)
    # 一个 data type 和数字的索引，比如 {'int': 1, 'flint', }
    mapping_list = {}
    for idx, mode in enumerate(mode_list):
        mapping_list[mode] = idx
        if idx == 0:
            compared_mse = quant_mse_list[mode]
        else:
            data_type_identify = torch.where(quant_mse_list[mode] < compared_mse, idx, data_type_identify)
            # update the compared_mse
            compared_mse = torch.where(quant_mse_list[mode] < compared_mse, quant_mse_list[mode], compared_mse)

    # 这个 mask 用于和 w_deq 点乘，取出 data type 对应的 w_deq
    data_type_mask = {}
    for mode in mode_list:
        data_type_mask[mode] = (data_type_identify == mapping_list[mode])

    w_deq = torch.zeros_like(w)
    if get_labels:
        codebook_set = torch.zeros((w.shape[0], 2 ** n_bit), device=w.device)
        labels_set = torch.zeros_like(w)
    for mode in mode_list:
        quant_grid_set[mode] = quant_grid_set[mode].to(data_type_mask[mode].device)
        w_deq = w_deq + torch.mul(w_deq_list[mode], data_type_mask[mode])
        if get_labels:
            labels_set = labels_set + torch.mul(labels_list[mode], data_type_mask[mode])
            # 将 quant_grid 进行维度拓展，和 data type 选择情况相乘，得到 codebook
            codebook_set = codebook_set + torch.mul(codebook_list[mode], data_type_mask[mode])

    mse = nn.MSELoss()
    w_deq = w_deq.half()

    if "group_outlier" in outlier_config['method']:
        w_deq = w_deq * ~outlier_mask + org_w * outlier_mask
    elif outlier_config['method'] == "olive_group":
        w_deq = w_deq * non_victim_mask
        w_deq = w_deq * ~outlier_mask + org_w * outlier_mask

    if outlier_config['method'] != "none":
        outlier_mask = outlier_mask.reshape(org_w_shape)
        non_victim_mask = non_victim_mask.reshape(org_w_shape)
    w_deq = w_deq.reshape(org_w_shape)
    org_w = org_w.reshape(org_w_shape)

    if "magnitude_weight" in outlier_config['method']:
        w_deq = w_deq * ~outlier_mask + org_w * outlier_mask
    elif outlier_config['method'] == "olive":
        w_deq = w_deq * non_victim_mask
        w_deq = w_deq * ~outlier_mask + org_w * outlier_mask


    # 打印统计信息
    if display:
        if outlier_config['method'] != "none":
            print(f"channel / group num:{w.shape[0]} alpha: {alpha} mse: {compared_mse.mean():.9f}, keep {outlier_config['keep_num']} outlier per group/channel")
        else:
            print(f"channel / group num:{w.shape[0]} alpha: {alpha} mse: {compared_mse.mean():.9f}, keep 0 outlier per group/channel")
        # overall_select = 0
        for mode in mode_list:
            overall_stats[mode] = overall_stats[mode].to(w.device)
            overall_stats[mode] = overall_stats[mode] + torch.count_nonzero(data_type_identify.view(-1) == mapping_list[mode]) / 1e5
            print(f"{mode}: {torch.count_nonzero(data_type_identify.view(-1) == mapping_list[mode])}")
    # exit(0)
    if get_labels:
        labels_set = labels_set.reshape(org_w_shape)
        return w_deq, labels_set, codebook_set
    else:
        return w_deq

@torch.no_grad()
def ant_quantization_search(w, n_bit, q_config, ant_config, outlier_config, outlier_normal_ratio=None, overall_stats={}, get_labels=False):

    best_mse = float('inf')
    best_alpha = 1.0
    lb, ub = ant_config['w_low'], ant_config['w_high']
    mse = nn.MSELoss()
    org_w_shape = w.shape

    # 防止 OOM，分为 N 个 slice 来量化
    slice_num = 4
    if get_labels:
        # codebook 数量由 group 数量决定
        codebook_num = w.shape[0] if q_config['q_group_size'] == -1 else w.shape[0] * w.shape[1] // q_config['q_group_size']
        codebook = torch.zeros((codebook_num, 2 ** n_bit), device=w.device)
        labels = torch.zeros_like(w, dtype=int)
        codebook = codebook.reshape(slice_num, -1)
        labels = labels.reshape(slice_num, -1)
    w = w.reshape(slice_num, -1)

    if outlier_config['method'] == 'olive':
        outlier_normal_ratio = outlier_ratio_stat(w, q_config, outlier_config['keep_ratio']).reshape(slice_num, -1)
    else:
        outlier_normal_ratio = torch.zeros_like(w)

    # 如果 w_low 和 w_high 之间的差距大于 10，进行 scale alpha 的搜索，alpha 的步长为 0.1
    if ant_config['w_high'] - ant_config['w_low'] > 10:
        for i in range(lb, ub, 10):
            search_alpha = i * 0.01
            quantize_mse = 0
            for slice_id, slice_w in enumerate(w):
                slice_w = slice_w.reshape(-1, org_w_shape[1])
                slice_ratio = outlier_normal_ratio[slice_id].reshape(-1, 1)

                # TODO: 支持 ant_asym 的情况
                slice_w_deq = ant_quantization(slice_w, n_bit=n_bit, **q_config, **ant_config, outlier_config=outlier_config, outlier_normal_ratio = slice_ratio, alpha=search_alpha, display=False)

                quantize_mse_slice = mse(slice_w, slice_w_deq)
                quantize_mse += quantize_mse_slice

            if quantize_mse < best_mse:
                best_mse, best_alpha = quantize_mse, search_alpha

    for slice_id, slice_w in enumerate(w):
        slice_w = slice_w.reshape(-1, org_w_shape[1])
        slice_ratio = outlier_normal_ratio[slice_id].reshape(-1, 1)

        # 正值和负值分别进行量化
        if ant_config['ant_asym']:
            zero_tensor = torch.zeros_like(slice_w)
            slice_w_pos = torch.where(slice_w > 0, slice_w, zero_tensor)
            slice_w_neg = torch.where(slice_w < 0, slice_w, zero_tensor)

            # 量化正值
            slice_w_deq_pos = ant_quantization(slice_w_pos, n_bit=n_bit, **q_config, **ant_config, outlier_config=outlier_config, outlier_normal_ratio=slice_ratio, alpha=best_alpha, display=True, overall_stats=overall_stats, pos_value=True)
            assert torch.min(slice_w_deq_pos) == 0

            # 量化负值
            slice_w_deq_neg = ant_quantization(slice_w_neg, n_bit=n_bit, **q_config, **ant_config, outlier_config=outlier_config, outlier_normal_ratio=slice_ratio, alpha=best_alpha, display=True, overall_stats=overall_stats, pos_value=False)

            assert torch.max(slice_w_deq_neg) == 0
            slice_w_deq = slice_w_deq_pos + slice_w_deq_neg
        else:
            if get_labels:
                slice_w_deq, slice_labels, slice_codebook = ant_quantization(slice_w, n_bit=n_bit, **q_config, **ant_config, outlier_config=outlier_config, outlier_normal_ratio=slice_ratio, alpha=best_alpha, display=True, overall_stats=overall_stats, get_labels=get_labels)
            else:
                slice_w_deq = ant_quantization(slice_w, n_bit=n_bit, **q_config, **ant_config, outlier_config=outlier_config, outlier_normal_ratio=slice_ratio, alpha=best_alpha, display=True, overall_stats=overall_stats)
    
        slice_w_deq = slice_w_deq.reshape(-1)
        w[slice_id] = slice_w_deq
        if get_labels:
            slice_codebook = slice_codebook.reshape(-1)
            slice_labels = slice_labels.reshape(-1)
            codebook[slice_id] = slice_codebook
            labels[slice_id] = slice_labels
    w = w.reshape(org_w_shape)
    if get_labels:
        labels = labels.reshape(codebook_num, -1)
        codebook = codebook.reshape(codebook_num, -1)

        return w, labels, codebook
    else:
        return w

if __name__ == '__main__':

    w = torch.normal(10, 5, size=(1024, 512))

    w_0 = w.detach().clone()
    w_1 = w.detach().clone()
    w_2 = w.detach().clone()
    w_3 = w.detach().clone()
    w_4 = w.detach().clone()

    mse = nn.MSELoss()

    outlier_config = {
    "method": "none",  
    "keep_ratio": -1.0,  
    "keep_num": 0, 
    }   

    # print_time("start")
    # ant_search(w_0, n_bit=4, ant_search_granularity=0, ant_mode="flint-int-meta_flint-pot")
    # print(f"tensor-wise mse: {mse(w, w_0)}")
    # print_time("time")

    # w_1 = ant_search(w_1, n_bit=4, ant_search_granularity=1, zero_point=True, ant_mode="flint-int-meta_flint-pot")
    # print(f"multi-channel: {mse(w, w_1)}")
    # print_time("time")

    # w_2 = ant_search(w_2, n_bit=4, ant_search_granularity=1, zero_point=False, ant_mode="flint-int-meta_flint-pot")
    # print(f"channel-wise: {mse(w, w_2)} ")
    # print_time("time")

    w_3 = ant_quantization(w_3, n_bit=6, zero_point=False, ant_mode="flint-int-float-nf", outlier_config=outlier_config, display=False)
    # print(f"channel-wise fast: {mse(w, w_3)} equal:{torch.eq(w_2, w_3)} {w_2} {w_3}")
    print(f"channel-wise fast: {mse(w, w_3)}")
    print_time("time")

    w_4 = w_4.cuda()
    centroids = torch.zeros(w_4.shape[0], 16).cuda()  
    initial_centroids = torch.zeros(w_4.shape[0], 16).cuda()

    labels = torch.zeros_like(w_4, dtype=torch.int32).cuda()
    output = torch.zeros_like(w_4).cuda()
    w_4 = kmeans_parallel.kmeans_cuda_forward(w_4, centroids, labels, output, initial_centroids, w_4.shape[0], w_4.shape[1], 16)
    print(f"kmeans: {mse(w.cuda(), w_4)} ")


    # w_4 = ant_quantization_search(w_4, n_bit=4, zero_point=False, ant_mode="flint-int-meta_flint-pot", ant_search_granularity=-1)
    # # print(f"channel-wise fast: {mse(w, w_4)} equal:{torch.eq(w_2, w_4)} {w_2} {w_4}")
    # print(f"channel-wise fast: {mse(w, w_4)} ")
    # print_time("time")

    # w_4 = ant_quantization(w_4, n_bit=4, ant_mode="flint-int-meta_flint-pot")
    # print(f"tensor-wise deprecated : {mse(w, w_4)}")
    # print_time("time")


