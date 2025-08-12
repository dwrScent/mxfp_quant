
import torch
import torch.nn as nn
from .ant_quant import generate_quant_grid
from .ant_quant import get_quant_weight
import torch.nn.functional as F
import math

from .quant_func import get_quant_mxfp

from .ant_quant import float_value, int_value, normal_float_value
from .rounding_comp import gemm_with_compensation_gpu

from .utils_stats import calculate_scale_range, calculate_outlier_exp
from ..utils.make_distribution import distri_3d

def extra_exp(tensor_value):
    tensor_exp = tensor_value.view(torch.int16)
    if tensor_value.dtype == torch.float16:
        return (tensor_exp >> 10) & 0x1F
    elif tensor_value.dtype == torch.bfloat16:
        return (tensor_exp >> 7) & 0xFF
    else:
        raise Exception(f"{tensor_value.dtype} not support yet")


def outlier_value(n_bit, signed=True, exp_bit=2, exp_base=5):
    B = n_bit - 1 if signed else n_bit
        
    value_bit = B
    mant_bit = value_bit - exp_bit
    values = []
    
    for i in range(exp_base, exp_base + 2 ** exp_bit):
        for j in range(int(2 ** mant_bit)):
            if i == exp_base and j == 0:
                continue

            v = 2 ** i * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if signed:
                values.append(-v)

    values = torch.tensor(values)
    values, _ = torch.sort(values)
                
    return values


@torch.no_grad()
def mxfp_sub_group(tensor_value, quant_grid, sub_group_grid, mode="int", zero_point=True, q_group_size=-1, sub_group_size=1, sub_group_mode='max', alpha=1.0, pos_value=None, get_labels=False, is_input=False, keep_outlier=False, print_stats=False):
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)
    sub_group_grid = sub_group_grid.to(tensor_value.device)

    # sub_group_size = 1

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)

    max_quant_val = max(quant_grid)
        
    # Compute the scaling factor
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp)

    # exp_max_val = torch.floor(torch.log2(max_val))
    # mask = torch.where(tensor_value > torch.pow(2, exp_max_val), torch.tensor(1), torch.tensor(0))

    zeros = 0

    # MODE: [max, outlier]
    # subgroup_mode = 'max'
    # subgroup_mode = 'outlier'

    if sub_group_mode == 'max':
        # find the sub group with maximum value
        outlier_mask = torch.zeros_like(tensor_value, dtype=torch.bool).to(tensor_value.device)
        _, indices = torch.topk(tensor_value.abs(), 1)
        outlier_mask.scatter_(1, indices, 1)

        outlier_group_mask = outlier_mask.reshape(-1, sub_group_size).to(dtype=torch.int8)
        outlier_group_mask = outlier_group_mask.sum(dim=1, keepdim=True)

        outlier_group_mask = outlier_group_mask.repeat(1, sub_group_size)
        outlier_group_mask = outlier_group_mask.reshape(-1, q_group_size)
    elif sub_group_mode == 'outlier':
        # labels = (((tensor_value + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        # tensor_q = quant_grid[labels]
        # assert torch.all(tensor_q < 8.0), "Tensor contains values greater than or equal to 8!"
        # outlier_mask = torch.where(tensor_q >= 4.0, 1.0, 0.).to(tensor_value.device)

        tensor_exp = extra_exp(tensor_value)
        max_val_exp = extra_exp(max_val)
        outlier_mask = (tensor_exp == max_val_exp)
        # print(tensor_exp, max_val_exp, outlier_mask, outlier_mask.shape)
        # exit(0)

        outlier_group_mask = outlier_mask.reshape(-1, sub_group_size).to(dtype=torch.int8)
        outlier_group_mask = outlier_group_mask.sum(dim=1, keepdim=True)
        # print(outlier_group_mask, outlier_mask.sum(dim=1, keepdim=True))
        outlier_group_mask = torch.where(outlier_group_mask >= 1, 1, 0)
        outlier_group_mask = outlier_group_mask.repeat(1, sub_group_size)
        outlier_group_mask = outlier_group_mask.reshape(-1, q_group_size)
    # print(outlier_group_mask.sum(), sub_group_grid, quant_grid)
    # exit(0)

    
    # print(outlier_mask, outlier_mask.shape, outlier_mask.sum(), outlier_group_mask, outlier_group_mask.shape, outlier_group_mask.sum())
    # exit(0)

    # Batch processing to avoid OOM
    batch_num = 4
    # batch_num = 16
    assert tensor_value.shape[0] % batch_num == 0
    batch_size = tensor_value.shape[0] // batch_num
    tensor_deq = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        tensor_q_par = quant_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par
    
    tensor_deq_o_group = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - sub_group_grid).abs().argmin(dim=-1)
        tensor_q_par = sub_group_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq_o_group[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    # print('init tensor_deq', (tensor_deq-tensor_value).abs().pow(2).to(torch.float32).mean())

    tensor_deq = tensor_deq * (1-outlier_group_mask) + tensor_deq_o_group * outlier_group_mask

    # tensor_deq = org_value * mask + tensor_deq * (1-mask)
    quant_mse = (tensor_deq-tensor_value).abs().pow(2).to(torch.float32)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    # calculate_max_error(tensor_value, tensor_deq, q_group_size=q_group_size)
    # print(quant_mse, quant_mse_sum, quant_mse_sum.mean())
    # print('merge tensor_deq', quant_mse_sum.mean())
    # exit(0)

    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(quant_mse).sum() == 0

    tensor_deq = tensor_deq.reshape(org_shape)

    quant_obj = 'input' if is_input else 'weight'
    if print_stats:
        print(f"Quantization MSE: {quant_mse_sum.mean().item()}, quant_obj: {quant_obj}, keep_outlier: {keep_outlier}, sub_gorup_size: {sub_group_size}")
    # print('init', scales, tensor_deq, tensor_value)

    if get_labels:
        return tensor_deq, quant_mse_sum, labels, quant_grid * scales
    else:
        return tensor_deq, quant_mse_sum
    
@torch.no_grad() 
def mxfp_sub_group_exscale(
        tensor_value,
        quant_grid,
        q_group_size=-1,
        sub_group_size=1,
        es_bit=2):
    """
    将输入 tensor_value 按 batch_num=4 分批次执行量化，
    调用 mxfp_sub_group_exscale_inner，最后把结果拼接回原形状。
    """
    batch_num = 4                      # 固定批次大小 
    dim0 = tensor_value.size(0) 
    
    # 保证可以被 batch_num 整除；若不能整除可改为向上取整并 pad 
    assert dim0 % batch_num == 0, f"dim0={dim0} must be divisible by batch_num={batch_num}"
    
    chunk_size = dim0 // batch_num 
    deq_list = []
    
    for i in range(batch_num):
        start = i * chunk_size 
        end   = start + chunk_size 
        sub_tensor = tensor_value[start:end]
        
        # 调用 inner 函数 
        deq_sub, _ = mxfp_sub_group_exscale_inner(
            sub_tensor,
            quant_grid,
            q_group_size=q_group_size,
            sub_group_size=sub_group_size,
            es_bit=es_bit 
        )
        
        deq_list.append(deq_sub) 
    
    # concat 回完整张量 
    tensor_deq      = torch.cat(deq_list,  dim=0)
    
    return tensor_deq, None



@torch.no_grad()
def mxfp_sub_group_exscale_inner(
        tensor_value,
        quant_grid,
        q_group_size=-1,
        sub_group_size=1,
        es_bit=2):
    """
    Two-level grouping:
      level-1: q_group_size   -> share one scale 
      level-2: sub_group_size -> pick one ratio (in {1,1.25,...}) that minimizes MSE 
    """
    assert torch.isnan(tensor_value).sum()  == 0 
    org_shape = tensor_value.shape  
    quant_grid = quant_grid.to(tensor_value.device) 
 
    # -------------------- level-1 grouping / scale --------------------
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0 
        tensor_value = tensor_value.reshape(-1,  q_group_size)
 
    max_val = tensor_value.abs().amax(dim=1,  keepdim=True)
    max_quant_val = quant_grid.max() 
 
    exp = torch.floor(torch.log2(max_val))  - torch.floor(torch.log2(max_quant_val))
    bias_mse = {}
    for bias in range(-1, 2):
        scales = torch.pow(2.,  exp+bias)                    # (n_group, 1)
        sub_groups_per_group = q_group_size // sub_group_size 
        scales = scales.expand(-1,  sub_groups_per_group).reshape(-1, 1)   # (n_subgrp_total,1)

        # -------------------- level-2 grouping --------------------
        if sub_group_size > 0:
            assert q_group_size % sub_group_size == 0 
            tensor_value = tensor_value.reshape(-1,  sub_group_size)       # (n_subgrp_total, sub_grp_size)

        ratios = torch.tensor([1.,  1.25, 1.5, 1.75] if es_bit == 2 else [1., 1.5],
                            dtype=torch.float16, 
                            device=tensor_value.device)                  # (n_ratio,)

        # --- for each sub-group: try every ratio and choose the best one ---
        # tensor_value : (n_subgrp_total , sub_grp_size)
        x_expanded = tensor_value.unsqueeze(2)                             # (n_subgrp_total , sub_grp_size , 1)
        scales_expanded = scales.unsqueeze(2)                              # (n_subgrp_total ,          , 1)

        cand_scales = scales_expanded * ratios.view(1,  1, -1)             # (n_subgrp_total ,          , n_ratio)
        cand_qval   = x_expanded / cand_scales                              # (n_subgrp_total , sub_grp_size , n_ratio)

        # nearest neighbor on grid 
        diff = cand_qval.unsqueeze(3)  - quant_grid.view(1,  1, 1, -1)      # (n_subgrp_total , sub_grp_size , n_ratio , n_grid)
        idx_best_qgrid = diff.abs().argmin(dim=3)                          # (n_subgrp_total , sub_grp_size , n_ratio)
        cand_dqval = torch.gather( 
            input=quant_grid.view(1,  1, 1, -1).expand_as(diff),   # shape same as diff 
            dim=3,
            index=idx_best_qgrid.unsqueeze(3)).squeeze(3)  * cand_scales   # (* , * , n_ratio)
        mse_per_ratio = (cand_dqval  - x_expanded).pow(2).mean(dim=1)   # (n_subgrp_total , n_ratio)

        best_ratio_idx = mse_per_ratio.argmin(dim=1)                       # (n_subgrp_total,)
        best_ratio     = ratios[best_ratio_idx]                             # (n_subgrp_total,)
        
        # gather final quantized-dequantized values 
        row_idx = torch.arange(tensor_value.size(0),  device=tensor_value.device) 
        best_dqval = cand_dqval[row_idx, :, best_ratio_idx]                 # (n_subgrp_total , sub_grp_size)
        # print(best_ratio_idx)

        # best_scales_final = scales.squeeze(1)  * best_ratio                # (n_subgrp_total,)
        
        # -------------------- statistics --------------------
        quant_mse_per_subgrp = mse_per_ratio[row_idx, best_ratio_idx]       # (n_subgrp_total,)

        # -------------------- reshape back --------------------
        tensor_deq = best_dqval.reshape(-1,  q_group_size)

        quant_mse_sum = quant_mse_per_subgrp.view(-1, 
                                            sub_groups_per_group).mean(dim=1,
                                                                        keepdim=True)
        bias_mse[bias] = (tensor_deq, quant_mse_sum)

    all_mse = torch.cat([bias_mse[b][1]  for b in range(-2,3)], dim=1)  
    # all_mse shape: [n_level1_group, n_bias=4]
    
    best_bias_idx = all_mse.argmin(dim=1)           # [n_level1_group]
    # print(best_bias_idx)

    all_deq = torch.stack([bias_mse[b][0]  for b in range(-2,3)], dim=0)  
    # [4, n_level1_group*q_group_size]
    all_deq = all_deq.view(3,-1,q_group_size)           # [4,n_level1,q_group_size]
    
    # repeat index to match last dim 
    idx_expanded = best_bias_idx.view(1,-1,1).expand(1,-1,q_group_size) 
    final_deq = torch.gather(all_deq,dim=0, 
                        index=idx_expanded).squeeze(0)
    
    tensor_deq = final_deq.reshape(org_shape).to(tensor_value.dtype) 
    
    # -------------------- sanity checks --------------------
    assert torch.isinf(tensor_deq).sum()  == 0 
    assert torch.isnan(tensor_deq).sum()  == 0 
 
    return tensor_deq, nn.functional.mse_loss(tensor_deq, tensor_value.reshape(org_shape))


@torch.no_grad() 
def mxfp_sub_group_exscale_direct(
        tensor_value,
        quant_grid,
        q_group_size=-1,
        sub_group_size=1,
        es_bit=2):
    """
    将输入 tensor_value 按 batch_num=4 分批次执行量化，
    调用 mxfp_sub_group_exscale_inner，最后把结果拼接回原形状。
    """
    batch_num = 4                      # 固定批次大小 
    dim0 = tensor_value.size(0) 
    
    # 保证可以被 batch_num 整除；若不能整除可改为向上取整并 pad 
    assert dim0 % batch_num == 0, f"dim0={dim0} must be divisible by batch_num={batch_num}"
    
    chunk_size = dim0 // batch_num 
    deq_list= []
    
    for i in range(batch_num):
        start = i * chunk_size 
        end   = start + chunk_size 
        sub_tensor = tensor_value[start:end]
        
        # 调用 inner 函数 
        deq_sub, _ = mxfp_sub_group_exscale_direct_inner(
            sub_tensor,
            quant_grid,
            q_group_size=q_group_size,
            sub_group_size=sub_group_size,
            es_bit=es_bit 
        )
        
        deq_list.append(deq_sub) 
    
    # concat 回完整张量 
    tensor_deq      = torch.cat(deq_list,  dim=0)
    
    return tensor_deq, None

@torch.no_grad()
def mxfp_sub_group_exscale_direct_inner(
        tensor_value,
        quant_grid,
        q_group_size=-1,
        sub_group_size=1,
        es_bit=2):
    """
    Two-level grouping:
      level-1: q_group_size   -> share one scale 
      level-2: sub_group_size -> pick one ratio (in {1,1.25,...}) that minimizes MSE 
    """
    assert torch.isnan(tensor_value).sum()  == 0 
    org_shape = tensor_value.shape  
    quant_grid = quant_grid.to(tensor_value.device) 
 
    # -------------------- level-1 grouping / scale --------------------
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0 
        tensor_value = tensor_value.reshape(-1,  q_group_size)
 
    max_val = tensor_value.abs().amax(dim=1,  keepdim=True)
    max_quant_val = quant_grid.max() 
 
    exp = torch.floor(torch.log2(max_val / 1.3125))  - torch.floor(torch.log2(max_quant_val))
    # for bias in range(-1, 1):
    scales = torch.pow(2.,  exp)                    # (n_group, 1)
    sub_groups_per_group = q_group_size // sub_group_size 
    scales = scales.expand(-1,  sub_groups_per_group).reshape(-1, 1)   # (n_subgrp_total,1)

    # -------------------- level-2 grouping --------------------
    if sub_group_size > 0:
        assert q_group_size % sub_group_size == 0 
        tensor_value = tensor_value.reshape(-1,  sub_group_size)       # (n_subgrp_total, sub_grp_size)

    ratios = torch.tensor([1.,  1.25, 1.5, 1.75] if es_bit == 2 else [1., 1.5],
                        dtype=torch.float16, 
                        device=tensor_value.device)                  # (n_ratio,)
    # ------------- new ratio selection by max-abs -------------
    qval = tensor_value / scales
    max_abs_per_subgrp = qval.abs().amax(dim=1,  keepdim=True)   # (n_subgrp_total , 1)
    target_ratio       = max_abs_per_subgrp / max_quant_val             # float 
    
    # ratios shape=(n_ratios,) → broadcast成 (n_subgrp_total , n_ratios)
    diff            = target_ratio - ratios.view(1,  -1)                  # (n_subgrp_total , n_ratios)
    best_ratio_idx  = diff.abs().argmin(dim=1)                            # (n_subgrp_total,)
    best_ratio      = ratios[best_ratio_idx]                             # same shape 
    # print(best_ratio_idx)
    
    # ------------- gather final dequantized values -------------
    row_idx        = torch.arange(tensor_value.size(0),  device=tensor_value.device) 
    cand_scales    = scales.squeeze(1)[row_idx]  * best_ratio             # (n_subgrp_total,)
    cand_scales    = cand_scales.view(-1,  1).expand(-1, sub_group_size)   # broadcast to each element 
    
    cand_qval      = tensor_value / cand_scales                          # (n_subgrp_total , sub_grp_size)
    diff_qgrid     = cand_qval.unsqueeze(2)  - quant_grid.view(1,  1, -1)   # (* , * , n_grid)
    idx_best_qgrid = diff_qgrid.abs().argmin(dim=2)                       # (* , *)
    best_dqval     = quant_grid[idx_best_qgrid] * cand_scales            # (* , *)
    # -------------------- reshape back --------------------
    tensor_deq = best_dqval.reshape(-1,  q_group_size)

    tensor_deq = tensor_deq.reshape(org_shape).to(tensor_value.dtype) 
    
    # -------------------- sanity checks --------------------
    assert torch.isinf(tensor_deq).sum()  == 0 
    assert torch.isnan(tensor_deq).sum()  == 0 
 
    return tensor_deq, nn.functional.mse_loss(tensor_deq, tensor_value.reshape(org_shape))

@torch.no_grad() 
def mxfp_sub_group_exscale_exp(
        tensor_value,
        quant_grid,
        q_group_size=-1,
        sub_group_size=1,
        ee_bit=2):
    """
    将输入 tensor_value 按 batch_num=4 分批次执行量化，
    调用 mxfp_sub_group_exscale_inner，最后把结果拼接回原形状。
    """
    batch_num = 4                      # 固定批次大小 
    dim0 = tensor_value.size(0) 
    
    # 保证可以被 batch_num 整除；若不能整除可改为向上取整并 pad 
    assert dim0 % batch_num == 0, f"dim0={dim0} must be divisible by batch_num={batch_num}"
    
    chunk_size = dim0 // batch_num 
    deq_list= []
    
    for i in range(batch_num):
        start = i * chunk_size 
        end   = start + chunk_size 
        sub_tensor = tensor_value[start:end]
        
        # 调用 inner 函数 
        deq_sub, _ = mxfp_sub_group_exscale_exp_inner(
            sub_tensor,
            quant_grid,
            q_group_size=q_group_size,
            sub_group_size=sub_group_size,
            ee_bit=ee_bit 
        )
        
        deq_list.append(deq_sub) 
    
    # concat 回完整张量 
    tensor_deq      = torch.cat(deq_list,  dim=0)
    
    return tensor_deq, None


@torch.no_grad()
def mxfp_sub_group_exscale_exp_inner(
        tensor_value,
        quant_grid,
        q_group_size=-1,
        sub_group_size=1,
        ee_bit=2):
    """
    Two-level grouping:
      level-1: q_group_size   -> share one scale 
      level-2: sub_group_size -> pick one ratio (in {1,1.25,...}) that minimizes MSE 
    """
    assert torch.isnan(tensor_value).sum()  == 0 
    org_shape = tensor_value.shape  
    quant_grid = quant_grid.to(tensor_value.device) 
 
    # -------------------- level-1 grouping / scale --------------------
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0 
        tensor_value = tensor_value.reshape(-1,  q_group_size)
 
    max_val = tensor_value.abs().amax(dim=1,  keepdim=True)
    max_quant_val = quant_grid.max() 
 
    exp = torch.floor(torch.log2(max_val))  - torch.floor(torch.log2(max_quant_val))
    bias_mse = {}
    range_ = range(-3, -1) if ee_bit == 2 else range(-1, 1)
    for bias in range_:
        scales = torch.pow(2.,  exp+bias)                    # (n_group, 1)
        sub_groups_per_group = q_group_size // sub_group_size 
        scales = scales.expand(-1,  sub_groups_per_group).reshape(-1, 1)   # (n_subgrp_total,1)

        # -------------------- level-2 grouping --------------------
        if sub_group_size > 0:
            assert q_group_size % sub_group_size == 0 
            tensor_value = tensor_value.reshape(-1,  sub_group_size)       # (n_subgrp_total, sub_grp_size)

        ratios = torch.tensor([1., 2, 4, 8] if ee_bit == 2 else [1., 2],
                            dtype=torch.float16, 
                            device=tensor_value.device)                  # (n_ratio,)

        # --- for each sub-group: try every ratio and choose the best one ---
        # tensor_value : (n_subgrp_total , sub_grp_size)
        x_expanded = tensor_value.unsqueeze(2)                             # (n_subgrp_total , sub_grp_size , 1)
        scales_expanded = scales.unsqueeze(2)                              # (n_subgrp_total ,          , 1)

        cand_scales = scales_expanded * ratios.view(1,  1, -1)             # (n_subgrp_total ,          , n_ratio)
        cand_qval   = x_expanded / cand_scales                              # (n_subgrp_total , sub_grp_size , n_ratio)

        # nearest neighbor on grid 
        diff = cand_qval.unsqueeze(3)  - quant_grid.view(1,  1, 1, -1)      # (n_subgrp_total , sub_grp_size , n_ratio , n_grid)
        idx_best_qgrid = diff.abs().argmin(dim=3)                          # (n_subgrp_total , sub_grp_size , n_ratio)
        cand_dqval = torch.gather( 
            input=quant_grid.view(1,  1, 1, -1).expand_as(diff),   # shape same as diff 
            dim=3,
            index=idx_best_qgrid.unsqueeze(3)).squeeze(3)  * cand_scales   # (* , * , n_ratio)
        mse_per_ratio = (cand_dqval  - x_expanded).pow(2).mean(dim=1)   # (n_subgrp_total , n_ratio)

        best_ratio_idx = mse_per_ratio.argmin(dim=1)                       # (n_subgrp_total,)
        best_ratio     = ratios[best_ratio_idx]                             # (n_subgrp_total,)
        
        # gather final quantized-dequantized values 
        row_idx = torch.arange(tensor_value.size(0),  device=tensor_value.device) 
        best_dqval = cand_dqval[row_idx, :, best_ratio_idx]                 # (n_subgrp_total , sub_grp_size)
        # print(best_ratio_idx)

        # best_scales_final = scales.squeeze(1)  * best_ratio                # (n_subgrp_total,)
        
        # -------------------- statistics --------------------
        quant_mse_per_subgrp = mse_per_ratio[row_idx, best_ratio_idx]       # (n_subgrp_total,)

        # -------------------- reshape back --------------------
        tensor_deq = best_dqval.reshape(-1,  q_group_size)

        quant_mse_sum = quant_mse_per_subgrp.view(-1, 
                                            sub_groups_per_group).mean(dim=1,
                                                                        keepdim=True)
        bias_mse[bias] = (tensor_deq, quant_mse_sum)

    all_mse = torch.cat([bias_mse[b][1]  for b in range_], dim=1)  
    # all_mse shape: [n_level1_group, n_bias=4]
    
    best_bias_idx = all_mse.argmin(dim=1)           # [n_level1_group]
    # print(best_bias_idx)

    all_deq = torch.stack([bias_mse[b][0]  for b in range_], dim=0)  
    # [4, n_level1_group*q_group_size]
    all_deq = all_deq.view(2,-1,q_group_size)           # [4,n_level1,q_group_size]
    
    # repeat index to match last dim 
    idx_expanded = best_bias_idx.view(1,-1,1).expand(1,-1,q_group_size) 
    final_deq = torch.gather(all_deq,dim=0, 
                        index=idx_expanded).squeeze(0)
    
    tensor_deq = final_deq.reshape(org_shape).to(tensor_value.dtype) 
    
    # -------------------- sanity checks --------------------
    assert torch.isinf(tensor_deq).sum()  == 0 
    assert torch.isnan(tensor_deq).sum()  == 0 
 
    return tensor_deq, nn.functional.mse_loss(tensor_deq, tensor_value.reshape(org_shape))

@torch.no_grad()
def sub_group_em(tensor_value, quant_grid, q_group_size=-1, sub_group_size=1,  get_labels=False, topk=1, em_bit=2):
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)
    sub_group_grid = float_value(4+em_bit, fix_e2b0=True)
    sub_group_grid = sub_group_grid.to(tensor_value.device)
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)

    max_quant_val = max(quant_grid)
        
    # Compute the scaling factor
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp)

    zeros = 0

    # Batch processing to avoid OOM
    batch_num = 4
    # batch_num = 16
    assert tensor_value.shape[0] % batch_num == 0
    batch_size = tensor_value.shape[0] // batch_num
    tensor_deq = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        tensor_q_par = quant_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par
    
    tmp = tensor_deq.reshape(-1, sub_group_size)
    outlier_mask = torch.zeros_like(tmp, dtype=torch.float16).to(tensor_value.device)

    _, indices = torch.topk(tmp.abs(), topk)
    outlier_mask.scatter_(1, indices, 1)
    outlier_group_mask = outlier_mask.reshape(-1, q_group_size)
    tensor_deq_o_group = torch.zeros_like(tensor_value)

    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - sub_group_grid).abs().argmin(dim=-1)
        tensor_q_par = sub_group_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq_o_group[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    tensor_deq = tensor_deq * (1-outlier_group_mask) + tensor_deq_o_group * outlier_group_mask

    quant_mse = (tensor_deq-tensor_value).abs().pow(2).to(torch.float32)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(quant_mse).sum() == 0

    tensor_deq = tensor_deq.reshape(org_shape)

    if get_labels:
        return tensor_deq, quant_mse_sum, labels, quant_grid * scales
    else:
        return tensor_deq, quant_mse_sum
    
def sub_group_em_real(tensor_value, quant_grid, q_group_size=-1, sub_group_size=1, topk=1, em_bit=2, fix=True):
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)
    sub_group_grid = float_value(4+em_bit, fix_e2b0=True)
    quant_grid, _ = quant_grid.sort(dim=0)
    sub_group_grid, _ = sub_group_grid.sort(dim=0)
    sub_group_grid = sub_group_grid.to(tensor_value.device)
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)

    max_quant_val = max(quant_grid)
        
    # Compute the scaling factor
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp)

    zeros = 0

    # Batch processing to avoid OOM
    batch_num = 1
    # batch_num = 16
    assert tensor_value.shape[0] % batch_num == 0
    batch_size = tensor_value.shape[0] // batch_num
    tensor_deq = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels1 = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        tensor_q_par = quant_grid[labels1] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par
    
    tmp = tensor_deq.reshape(-1, sub_group_size)
    outlier_mask1 = torch.zeros_like(tmp, dtype=torch.int).to(tensor_value.device)

    _, indices = torch.topk(tmp.abs(), topk)
    k_rank = torch.arange(1  , topk+1 ,
                      dtype=torch.int  ,
                      device=tensor_value.device).expand_as(indices)
    outlier_mask1.scatter_(dim=1  , index=indices , src=k_rank)

    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels2 = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - sub_group_grid).abs().argmin(dim=-1)

    labels2 = labels2.reshape(-1, sub_group_size)
    inf = torch.tensor(float('inf'),  device=outlier_mask1.device) 
    mask_for_sort = torch.where(outlier_mask1  == 0, inf, outlier_mask1.float()) 
    sorted_idx = torch.argsort(mask_for_sort,  dim=-1)[:, :topk]
    em_info = torch.gather(labels2, 
                     dim=-1,
                     index=sorted_idx)
    stride = int(2**em_bit)
    em_info_high = torch.where(em_info >= int(2 **(3+em_bit)), em_info // stride, (em_info - (stride - 1)) // stride)
    em_info_low = em_info - torch.where(em_info_high >= 8, stride * em_info_high, stride * em_info_high + stride - 1)
    
    labels = labels1.reshape(-1, sub_group_size)
    labels = labels * (1 - (outlier_mask1 != 0).to(torch.int))
    rank_to_val = em_info_high.gather(            # [B*H*W , sub_group_size]
        dim=1,
        index=(outlier_mask1.to(torch.int64) - 1).clamp(min=0)   # mask==0 → -1 → clamp→0 
    )
    labels.scatter_add_( 
        dim=1,
        index=torch.arange(sub_group_size, 
                           device=outlier_mask1.device).expand_as(outlier_mask1), 
        src=rank_to_val * (outlier_mask1 > 0).to(rank_to_val.dtype) 
    )

    labels = labels.reshape(tensor_value.shape).to(torch.int)
    for idx in range(batch_num):
        tensor_q_par = quant_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par
    
    tmp = tensor_deq.reshape(-1, sub_group_size)
    outlier_mask2 = torch.zeros_like(tmp, dtype=torch.int).to(tensor_value.device)

    _, indices = torch.topk(tmp.abs(), topk)
    k_rank = torch.arange(1  , topk+1 ,
                      dtype=torch.int  ,
                      device=tensor_value.device).expand_as(indices)
    outlier_mask2.scatter_(dim=1  , index=indices , src=k_rank)

    if fix:
        the_same = (outlier_mask1 == outlier_mask2).reshape(-1, sub_group_size).to(torch.float).amin(-1, keepdim=True)
        outlier_mask = outlier_mask1 * the_same
    else:
        outlier_mask = outlier_mask1
    labels = labels1.reshape(-1, sub_group_size)
    labels = (labels * (1 - (outlier_mask != 0).to(torch.float))).to(em_info_high.dtype)
    rank_to_val = em_info_high.gather(            # [B*H*W , sub_group_size]
        dim=1,
        index=(outlier_mask.to(torch.int64) - 1).clamp(min=0)   # mask==0 → -1 → clamp→0 
    )
    labels.scatter_add_( 
        dim=1,
        index=torch.arange(sub_group_size, 
                           device=outlier_mask.device).expand_as(outlier_mask), 
        src=rank_to_val * (outlier_mask > 0).to(rank_to_val.dtype) 
    )
    
    labels = labels.reshape(-1, sub_group_size)
    labels = torch.where(labels >= 8, stride * labels, stride * labels + stride - 1)
    rank_to_val = em_info_low.gather(            # [B*H*W , sub_group_size]
        dim=1,
        index=(outlier_mask.to(torch.int64) - 1).clamp(min=0)   # mask==0 → -1 → clamp→0 
    )
    labels.scatter_add_( 
        dim=1,
        index=torch.arange(sub_group_size, 
                           device=outlier_mask.device).expand_as(outlier_mask), 
        src=rank_to_val * (outlier_mask > 0).to(rank_to_val.dtype) 
    )
    err = (labels[outlier_mask != 0] != labels2[outlier_mask != 0]).to(torch.float).sum()
    assert err == 0, err
    labels = labels.reshape(tensor_value.shape).to(torch.int)
    for idx in range(batch_num):
        tensor_q_par = sub_group_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    quant_mse = (tensor_deq-tensor_value).abs().pow(2).to(torch.float32)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(quant_mse).sum() == 0

    tensor_deq = tensor_deq.reshape(org_shape)

    return tensor_deq, quant_mse_sum

@torch.no_grad()
def mxfp_sub_group_heuristic(tensor_value, quant_grid, sub_group_grid, mode="int", zero_point=True,
                             q_group_size=-1, sub_group_size=1, sub_group_mode=None, alpha=1.0, pos_value=None,
                             get_labels=False, is_input=False, keep_outlier=False, print_stats=False):
    org_shape = tensor_value.shape

    # ant_mode = 'float-int-pot'
    # ant_mode = 'float-int'
    ant_mode = sub_group_mode
    mode_list = ant_mode.split('-')
    quant_grid_set = generate_quant_grid(n_bit=4, signed=True, ant_mode=ant_mode)

    w_deq_list = {}
    for mode in mode_list:
        w_deq_list[mode], _ = get_quant_mxfp(
            tensor_value, quant_grid_set[mode],
            q_group_size=q_group_size,
            keep_outlier=keep_outlier,
            print_stats=print_stats
        )

    # 提取 exponent
    tensor_exponent = extra_exp(tensor_value)
    tensor_exponent = tensor_exponent.reshape(-1, sub_group_size)

    tensor_reshaped_for_max = tensor_value.reshape(-1, q_group_size)
    max_val = tensor_reshaped_for_max.abs().amax(dim=1, keepdim=True)
    max_val_exponent = extra_exp(max_val)
    num_sub_groups_per_group = q_group_size // sub_group_size
    max_val_exponent_expanded = max_val_exponent.repeat_interleave(num_sub_groups_per_group, dim=0)

    # Rule 1: INT mask
    exact_match_mask = (tensor_exponent == max_val_exponent_expanded)
    int_mask = torch.any(exact_match_mask, dim=1)

    # Initialize mask dict
    data_type_mask = {'int': int_mask}

    if 'pot' in mode_list:
        # Rule 2: POT mask — all elements' exponent diff > 3
        exponent_diff = (tensor_exponent - max_val_exponent_expanded).abs()
        pot_mask = torch.all(exponent_diff > 3, dim=1) & (~int_mask)
        data_type_mask['pot'] = pot_mask
        float_mask = ~(int_mask | pot_mask)
    else:
        float_mask = ~int_mask

    if 'float' in mode_list:
        data_type_mask['float'] = float_mask

    # 反量化 & 合并
    tensor_deq = torch.zeros_like(tensor_value).reshape(-1, sub_group_size)
    for mode in mode_list:
        w_deq_mode = w_deq_list[mode].reshape(-1, sub_group_size)
        mask_mode = data_type_mask[mode]
        if mask_mode.any():
            tensor_deq[mask_mode] = w_deq_mode[mask_mode]

    tensor_deq = tensor_deq.reshape(org_shape)
    quant_mse = nn.MSELoss()(tensor_value, tensor_deq)

    return tensor_deq, quant_mse
    
@torch.no_grad()
def mxfp_sub_group_adaptive(tensor_value, quant_grid, sub_group_grid, mode="int", zero_point=True, q_group_size=-1, sub_group_size=1, sub_group_mode=None, alpha=1.0, pos_value=None, get_labels=False, is_input=False, keep_outlier=False, print_stats=False):
    org_shape = tensor_value.shape
    ant_mode = 'float-int-pot'
    # ant_mode = 'float-int'
    # ant_mode = sub_group_mode
    mode_list = ant_mode.split('-')
    quant_grid_set = generate_quant_grid(n_bit=4, signed=True, ant_mode=ant_mode)
    w_deq_list = {}
    quant_mse_list = {}

    # sub_group_size = 4

    for mode in mode_list:
        w_deq_list[mode], _ = get_quant_mxfp(tensor_value, quant_grid_set[mode], q_group_size=q_group_size, keep_outlier=keep_outlier, print_stats=print_stats)
        exist_mode = mode

        quant_mse = (w_deq_list[mode]-tensor_value).abs().pow(2).to(torch.float32)
        quant_mse = quant_mse.reshape(-1, sub_group_size)
        quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)
        quant_mse_list[mode] = quant_mse_sum
    
    if sub_group_size > 0:
        tensor_value = tensor_value.reshape(-1, sub_group_size)
        for mode in mode_list:
            w_deq_list[mode] = w_deq_list[mode].reshape(-1, sub_group_size)
    
    data_type_identify = torch.zeros_like(quant_mse_list[exist_mode], dtype=torch.int32)
    mapping_list = {}
    for idx, mode in enumerate(mode_list):
        mapping_list[mode] = idx
        if idx == 0:
            compared_mse = quant_mse_list[mode]
        else:
            data_type_identify = torch.where(quant_mse_list[mode] < compared_mse, idx, data_type_identify)
            # update the compared_mse
            compared_mse = torch.where(quant_mse_list[mode] < compared_mse, quant_mse_list[mode], compared_mse)
    data_type_mask = {}
    for mode in mode_list:
        data_type_mask[mode] = (data_type_identify == mapping_list[mode])
    
    # print(data_type_mask['int'].shape, w_deq_list['int'].shape, data_type_identify.shape, quant_mse_list['int'].shape)

    tensor_deq = torch.zeros_like(tensor_value, dtype=torch.float16)
    for mode in mode_list:
        quant_grid_set[mode] = quant_grid_set[mode].to(data_type_mask[mode].device)
        tensor_deq = tensor_deq + torch.mul(w_deq_list[mode], data_type_mask[mode])

    mse = nn.MSELoss()
    quant_mse = mse(tensor_value, tensor_deq)
    tensor_deq = tensor_deq.reshape(org_shape)

    return tensor_deq, quant_mse

@torch.no_grad()
def mxfp_sub_group_adaptive_em(tensor_value, quant_grid, sub_group_grid, mode="int", zero_point=True, q_group_size=-1, sub_group_size=1, sub_group_mode=None, alpha=1.0, pos_value=None, get_labels=False, is_input=False, keep_outlier=False, print_stats=False):
    org_shape = tensor_value.shape
    # ant_mode = 'float-int-pot'
    ant_mode = 'float-int'
    # ant_mode = sub_group_mode
    mode_list = ant_mode.split('-')
    quant_grid_set = generate_quant_grid(n_bit=4, signed=True, ant_mode=ant_mode)
    w_deq_list = {}
    quant_mse_list = {}

    max_val = tensor_value.reshape(-1, q_group_size).abs().amax(dim=1, keepdim=True)

    # Calculate the exponent for the tensor and max_val
    tensor_exp = extra_exp(tensor_value.reshape(-1, q_group_size))
    max_val_exp = extra_exp(max_val)
    outlier_mask = (tensor_exp == max_val_exp)

    # Find the outlier (exp == max exp) of each subgroup
    num_sub_groups = tensor_value.numel() // sub_group_size
    reshaped_outlier_mask = outlier_mask.reshape(num_sub_groups, sub_group_size)
    reshaped_tensor_abs = tensor_value.abs().reshape(num_sub_groups, sub_group_size)
    potential_outlier_values = reshaped_tensor_abs * reshaped_outlier_mask
    max_outlier_val_in_subgroup = torch.max(potential_outlier_values, dim=1, keepdim=True)[0]
    final_element_wise_mask_reshaped = (potential_outlier_values == max_outlier_val_in_subgroup) & (reshaped_outlier_mask)
    outlier_group_mask = final_element_wise_mask_reshaped.reshape(-1, q_group_size).to(tensor_value.dtype)

    # TODO: get the subgroup grid based on the ant_mode
    all_sub_group_grids = {
        'float': [0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
        'int': [0, -4.0, -4.25, -4.5, -4.75, -5.0, -5.25, -5.5, -5.75,-6.0, -6.25, -6.5, -6.75, -7.0, -7.25, -7.5, -7.75,4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75,6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75],
        # 'pot': [0, -4.0, -5.0, -6.0, -7.0, 4.0, 5.0, 6.0, 7.0]
    }
    sub_group_grid_set = {}
    for mode in mode_list:
        if mode in all_sub_group_grids:
            sub_group_grid_set[mode] = torch.tensor(
                all_sub_group_grids[mode],
                dtype=tensor_value.dtype,
                device=tensor_value.device
            )

    for mode in mode_list:
        w_deq_list[mode], _ = get_quant_mxfp(tensor_value, quant_grid_set[mode], q_group_size=q_group_size, keep_outlier=keep_outlier, print_stats=print_stats)
        exist_mode = mode

        # TODO: add extra mantissa for the subgroup based on the sub_group_grid
        sub_group_grid = sub_group_grid_set[mode]
        w_deq_subgroup, _ = get_quant_mxfp(tensor_value, sub_group_grid, q_group_size=q_group_size, keep_outlier=keep_outlier, print_stats=print_stats)

        outlier_group_mask = outlier_group_mask.reshape(org_shape)
        w_deq_list[mode] = torch.where(outlier_group_mask.bool(),w_deq_subgroup,  w_deq_list[mode])

        # Calculate the quantization MSE
        quant_mse = (w_deq_list[mode]-tensor_value).abs().pow(2).to(torch.float32)
        quant_mse = quant_mse.reshape(-1, sub_group_size)
        quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)
        quant_mse_list[mode] = quant_mse_sum
    
    if sub_group_size > 0:
        tensor_value = tensor_value.reshape(-1, sub_group_size)
        for mode in mode_list:
            w_deq_list[mode] = w_deq_list[mode].reshape(-1, sub_group_size)
    
    data_type_identify = torch.zeros_like(quant_mse_list[exist_mode], dtype=torch.int32)
    mapping_list = {}
    for idx, mode in enumerate(mode_list):
        mapping_list[mode] = idx
        if idx == 0:
            compared_mse = quant_mse_list[mode]
        else:
            data_type_identify = torch.where(quant_mse_list[mode] < compared_mse, idx, data_type_identify)
            # update the compared_mse
            compared_mse = torch.where(quant_mse_list[mode] < compared_mse, quant_mse_list[mode], compared_mse)
    data_type_mask = {}
    for mode in mode_list:
        data_type_mask[mode] = (data_type_identify == mapping_list[mode])

    tensor_deq = torch.zeros_like(tensor_value, dtype=torch.float16)
    for mode in mode_list:
        quant_grid_set[mode] = quant_grid_set[mode].to(data_type_mask[mode].device)
        tensor_deq = tensor_deq + torch.mul(w_deq_list[mode], data_type_mask[mode])

    mse = nn.MSELoss()
    quant_mse = mse(tensor_value, tensor_deq)
    tensor_deq = tensor_deq.reshape(org_shape)
    if print_stats:
        print("-" * 50)
        print("Adaptive Data Type Selection Statistics:")
        total_sub_groups = data_type_identify.numel()
        for mode, idx in mapping_list.items():
            count = (data_type_identify == idx).sum().item()
            percentage = (count / total_sub_groups) * 100 if total_sub_groups > 0 else 0
            print(f"  - Mode '{mode}': Chosen for {count}/{total_sub_groups} sub-groups ({percentage:.2f}%)")
        print("-" * 50)

    return tensor_deq, quant_mse

@torch.no_grad()
def mxfp_sub_group_heuristic_em(tensor_value, quant_grid, sub_group_grid, mode="int", zero_point=True, q_group_size=-1, sub_group_size=1, sub_group_mode=None, alpha=1.0, pos_value=None, get_labels=False, is_input=False, keep_outlier=False, print_stats=False):
    """
    Applies quantization to a tensor using a heuristic approach to select the data type
    for each sub-group based on outlier characteristics.

    The four heuristic rules are:
    1.  >=2 outliers: E1M2 (int) quantization + extra mantissa for the largest outlier.
    2.  1 outlier:     E2M1 (float) quantization + extra mantissa for the outlier.
    3.  All subnormal: E3M0 (pot) quantization, no extra mantissa.
    4.  Others:        E2M1 (float) quantization, no extra mantissa.
    """
    org_shape = tensor_value.shape
    device = tensor_value.device
    dtype = tensor_value.dtype

    ant_mode = 'float-int-pot'
    # 1. DEFINE QUANTIZATION GRIDS
    quant_grid_set = generate_quant_grid(n_bit=4, signed=True, ant_mode=ant_mode)
    for mode_key in quant_grid_set:
        quant_grid_set[mode_key] = quant_grid_set[mode_key].to(device)
    all_sub_group_grids = {
        'float': [0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
        'int': [0, -4.0, -4.25, -4.5, -4.75, -5.0, -5.25, -5.5, -5.75,-6.0, -6.25, -6.5, -6.75, -7.0, -7.25, -7.5, -7.75,4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75,6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75],
    }
    sub_group_grid_set = {
        mode_key: torch.tensor(grid, dtype=dtype, device=device)
        for mode_key, grid in all_sub_group_grids.items()
    }

    # 2. GENERATE HEURISTIC MASKS
    tensor_reshaped = tensor_value.reshape(-1, q_group_size)
    max_val = tensor_reshaped.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)

    tensor_exponent = extra_exp(tensor_value)
    tensor_exponent = tensor_exponent.reshape(-1, sub_group_size)
    
    num_sub_groups = tensor_value.numel() // sub_group_size
    max_val_exponent = extra_exp(max_val)
    num_sub_groups_per_group = q_group_size // sub_group_size
    max_val_exponent_expanded = max_val_exponent.repeat_interleave(num_sub_groups_per_group, dim=0)

    exact_match_mask = (tensor_exponent == max_val_exponent_expanded)
    outlier_counts = exact_match_mask.sum(dim=1)

    mask_ge2_outliers = (outlier_counts >= 2)
    mask_1_outlier = (outlier_counts == 1)
    exponent_diff = (tensor_exponent - max_val_exponent_expanded).abs()
    mask_subnormal = torch.all(exponent_diff > 3, dim=1) & (outlier_counts == 0)
    mask_others = ~(mask_ge2_outliers | mask_1_outlier | mask_subnormal)

    heuristic_masks = {
        'ge2_outliers': mask_ge2_outliers,
        '1_outlier': mask_1_outlier,
        'subnormal': mask_subnormal,
        'others': mask_others
    }

    # 3. PERFORM QUANTIZATION FOR EACH OF THE 4 STRATEGIES
    
    # --- Helper function for batch processing to prevent OOM ---
    def _batch_quantize_dequantize(input_tensor, scales, grid, batch_num=4):
        assert input_tensor.shape[0] % batch_num == 0, \
            f"Batch dimension mismatch! Tensor shape[0] ({input_tensor.shape[0]}) must be divisible by batch_num ({batch_num})"
        
        batch_size = input_tensor.shape[0] // batch_num
        output_deq = torch.zeros_like(input_tensor)

        for i in range(batch_num):
            start_idx, end_idx = i * batch_size, (i + 1) * batch_size
            tensor_batch = input_tensor[start_idx:end_idx, :]
            scales_batch = scales[start_idx:end_idx, :]
            
            labels = ((tensor_batch / scales_batch).unsqueeze(-1) - grid).abs().argmin(dim=-1)
            deq_batch = grid[labels] * scales_batch
            output_deq[start_idx:end_idx, :] = deq_batch
        return output_deq

    # --- Strategy 1: >=2 outliers (E1M2 + extra mantissa) ---
    reshaped_tensor_abs = tensor_value.abs().reshape(num_sub_groups, sub_group_size)
    potential_outliers = reshaped_tensor_abs * exact_match_mask.reshape(num_sub_groups, sub_group_size)
    max_outlier_val, _ = torch.max(potential_outliers, dim=1, keepdim=True)
    outlier_final_mask_reshaped = (potential_outliers == max_outlier_val) & (exact_match_mask)
    outlier_final_mask = outlier_final_mask_reshaped.reshape(-1, q_group_size).to(dtype)

    scales_int = torch.pow(2, torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(sub_group_grid_set['int'].abs().amax())))
    
    # ** MODIFICATION: Using batch processing **
    deq_std_int = _batch_quantize_dequantize(tensor_reshaped, scales_int, quant_grid_set['int'])
    deq_sub_int = _batch_quantize_dequantize(tensor_reshaped, scales_int, sub_group_grid_set['int'])
    deq1 = torch.where(outlier_final_mask.bool(), deq_sub_int, deq_std_int)

    del deq_std_int, deq_sub_int

    # --- Strategy 2: 1 outlier (E2M1 + extra mantissa) ---
    scales_float = torch.pow(2, torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(sub_group_grid_set['float'].abs().amax())))

    # ** MODIFICATION: Using batch processing **
    deq_std_float = _batch_quantize_dequantize(tensor_reshaped, scales_float, quant_grid_set['float'])
    deq_sub_float = _batch_quantize_dequantize(tensor_reshaped, scales_float, sub_group_grid_set['float'])
    deq2 = torch.where(outlier_final_mask.bool(), deq_sub_float, deq_std_float)

    del deq_std_float, deq_sub_float
    
    # --- Strategy 3: All subnormal (E3M0, no extra mantissa) ---
    deq3, _ = get_quant_mxfp(tensor_value, quant_grid_set['pot'], q_group_size=q_group_size)
    deq3 = deq3.reshape(-1, q_group_size)

    # --- Strategy 4: Others (E2M1, no extra mantissa) ---
    deq4, _ = get_quant_mxfp(tensor_value, quant_grid_set['float'], q_group_size=q_group_size)
    deq4 = deq4.reshape(-1, q_group_size)

    # 4. COMBINE RESULTS USING MASKS
    tensor_deq = torch.zeros_like(tensor_reshaped)
    for i, deq_tensor in enumerate([deq1, deq2, deq3, deq4], 1):
        mask_key = list(heuristic_masks.keys())[i-1]
        if heuristic_masks[mask_key].any():
            expanded_mask = heuristic_masks[mask_key].unsqueeze(-1).expand(-1, sub_group_size).reshape(-1, q_group_size)
            tensor_deq += deq_tensor * expanded_mask

    # 5. FINAL MSE AND STATISTICS
    quant_mse = nn.MSELoss()(tensor_reshaped, tensor_deq)
    tensor_deq = tensor_deq.reshape(org_shape)

    if print_stats:
        print("-" * 50)
        print("Heuristic Data Type Selection Statistics:")
        total_sub_groups = num_sub_groups
        for name, mask in heuristic_masks.items():
            count = mask.sum().item()
            percentage = (count / total_sub_groups) * 100 if total_sub_groups > 0 else 0
            print(f"  - Rule '{name}': Applied to {count}/{total_sub_groups} sub-groups ({percentage:.2f}%)")
        print("-" * 50)

    return tensor_deq, quant_mse

class MXFP_Linear(nn.Module):
    def __init__(self, w_bit, a_bit, group_size, in_features, out_features, bias, dev, ant_config, layer_id, layer_name):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.group_size = group_size 
        self.ant_config = ant_config

        self.layer_id = layer_id
        self.layer_name = layer_name

        # ANT param
        self.weight_quant_grid = None
        self.weight_alpha = -1
        self.input_quant_grid = None
        self.input_alpha = -1

        self.search_tag = None
        self.keep_outlier = False
        # self.keep_outlier = True

        self.print_stats = False
        # self.print_stats = True

        # MXFP param
        self.weight_mxfp_mode = ant_config['weight_mxfp_mode']
        self.input_mxfp_mode = ant_config['input_mxfp_mode']

        self.weight_sub_group_size = ant_config.get('weight_sub_group_size')
        self.weight_sub_group_mode = ant_config.get('weight_sub_group_mode')
        self.input_sub_group_size = ant_config.get('input_sub_group_size')
        self.input_sub_group_mode = ant_config.get('input_sub_group_mode')
        self.topk = ant_config.get("topk")
        self.em_bit = ant_config.get("em_bit")
        self.es_bit = ant_config.get("es_bit")
        self.ee_bit = ant_config.get("ee_bit")
        self.fix = ant_config.get("fix")

        assert self.in_features % self.group_size == 0

        self.exp_bit_width = None

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, a_bit, group_size, layer_id, layer_name, init_only=False, ant_config=None, quant_mode=None):

        in_features = linear.weight.shape[1] 
        out_features = linear.weight.shape[0]

        mxfp_linear = cls(w_bit, a_bit, group_size, in_features, out_features, linear.bias is not None, linear.weight.device, ant_config, layer_id, layer_name)
        if init_only:  # just prepare for loading sd
            return mxfp_linear

        # mxfp_linear.weight = linear.weight.data.clone().half()
        mxfp_linear.weight = linear.weight.data
        
        if linear.bias is not None:
            mxfp_linear.bias = linear.bias.clone().half()


        # w_exp_field_map = {3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
        # w_exp_field = w_exp_field_map[w_bit]
        flint_r_list = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5]

        if w_bit < 16:
            if ant_config['ant_mode'] == 'int':
                mxfp_linear.weight_quant_grid = int_value(w_bit, True)
            elif ant_config['ant_mode'] == 'float':
                mxfp_linear.weight_quant_grid = float_value(w_bit, True)
            else:
                raise NotImplementedError('Not support yet.')
            # mxfp_linear.weight_quant_grid = torch.tensor(flint_r_list)
        # mxfp_linear.weight_quant_grid = normal_float_value(w_bit, True)

        # a_exp_field_map = {3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
        # a_exp_field = a_exp_field_map[a_bit]
        if a_bit < 16:
            if ant_config['ant_mode'] == 'int':
                mxfp_linear.input_quant_grid = int_value(a_bit, True)
            elif ant_config['ant_mode'] == 'float':
                mxfp_linear.input_quant_grid = float_value(a_bit, True)
            else:
                raise NotImplementedError('Not support yet.')
            # mxfp_linear.input_quant_grid = torch.tensor(flint_r_list)
        # mxfp_linear.input_quant_grid = normal_float_value(a_bit, True)

        assert mxfp_linear.group_size == 32
            
        return mxfp_linear
    
    def _quantize_data(self, data, mode, quant_grid, n_bit, exp_base, is_input, sub_group_size, sub_group_mode):
        # sub group with E0M3
        sub_group_grid = [0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        # sub_group_grid = [0, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -7.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]
        sub_group_grid = torch.tensor(sub_group_grid)    
        quantize_methods = {
            'base': lambda: get_quant_mxfp(data, quant_grid=quant_grid, mode=None, zero_point=False, q_group_size=self.group_size, is_input=is_input, keep_outlier=self.keep_outlier, print_stats=self.print_stats),
            'sub_group': lambda: mxfp_sub_group(data, quant_grid=quant_grid, sub_group_grid=sub_group_grid, mode=None, zero_point=False, q_group_size=self.group_size, sub_group_size=sub_group_size, sub_group_mode=sub_group_mode, print_stats=self.print_stats),
            'sub_group_adaptive': lambda: mxfp_sub_group_adaptive(data, quant_grid=quant_grid, sub_group_grid=sub_group_grid, mode=None, zero_point=False, q_group_size=self.group_size, sub_group_size=sub_group_size, sub_group_mode=sub_group_mode, print_stats=self.print_stats),
            'sub_group_heuristic': lambda: mxfp_sub_group_heuristic(data, quant_grid=quant_grid, sub_group_grid=sub_group_grid, mode=None, zero_point=False, q_group_size=self.group_size, sub_group_size=sub_group_size, sub_group_mode=sub_group_mode, print_stats=self.print_stats),
            'sub_group_em': lambda: sub_group_em(data, quant_grid=quant_grid, q_group_size=self.group_size, sub_group_size=sub_group_size, topk=self.topk, em_bit=self.em_bit),
            'sub_group_em_real': lambda: sub_group_em_real(data, quant_grid=quant_grid, q_group_size=self.group_size, sub_group_size=sub_group_size, topk=self.topk, em_bit=self.em_bit, fix=self.fix),
            'sub_group_es': lambda: mxfp_sub_group_exscale(data, quant_grid=quant_grid, q_group_size=self.group_size, sub_group_size=sub_group_size, es_bit=self.es_bit),
            'sub_group_es_direct': lambda: mxfp_sub_group_exscale_direct(data, quant_grid=quant_grid, q_group_size=self.group_size, sub_group_size=sub_group_size, es_bit=self.es_bit),
            'sub_group_ee': lambda: mxfp_sub_group_exscale_exp(data, quant_grid=quant_grid, q_group_size=self.group_size, sub_group_size=sub_group_size, ee_bit=self.ee_bit),
            'sub_group_adaptive_em': lambda: mxfp_sub_group_adaptive_em(data, quant_grid=quant_grid, sub_group_grid=sub_group_grid, mode=None, zero_point=False, q_group_size=self.group_size, sub_group_size=sub_group_size, sub_group_mode=sub_group_mode, print_stats=self.print_stats),
            'sub_group_heuristic_em': lambda: mxfp_sub_group_heuristic_em(data, quant_grid=quant_grid, sub_group_grid=sub_group_grid, mode=None, zero_point=False, q_group_size=self.group_size, sub_group_size=sub_group_size, sub_group_mode=sub_group_mode, print_stats=self.print_stats),
        }
        return quantize_methods.get(mode, lambda: NotImplementedError(f'not support this mxfp mode: {mode}'))()

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input = x.reshape(-1, x.shape[-1])

        # Search and set data type and alpha in the first inference
        if self.search_tag is None:
            # calculate_scale_range(self.weight, self.weight_quant_grid, self.layer_id, self.layer_name, self.group_size, False)
            # calculate_outlier_exp(self.weight, self.weight_quant_grid, self.layer_id, self.layer_name, self.group_size, False)

            # for weight
            # distri_3d(self.weight, layer_idx=self.layer_id, layer_name=self.layer_name)

            if self.w_bit < 16:
                deq_weight, _ = self._quantize_data(self.weight, self.weight_mxfp_mode, self.weight_quant_grid, self.w_bit, 5, False, self.weight_sub_group_size, self.weight_sub_group_mode)
            else:
                deq_weight = self.weight
            
            # Quantize weight only once
            self.weight = deq_weight

            if self.a_bit < 16:
                deq_input, _ = self._quantize_data(input, self.input_mxfp_mode, self.input_quant_grid, self.a_bit, 7, True, self.input_sub_group_size, self.input_sub_group_mode)
            else:
                deq_input = input
                
            self.search_tag = 1

        # quantize input based on the selected data type and alpha
        else:
            if self.a_bit < 16:
                # calculate_scale_range(input, self.input_quant_grid, self.layer_id, self.layer_name, self.group_size, True)

                deq_input, _ = self._quantize_data(input, self.input_mxfp_mode, self.input_quant_grid, self.a_bit, 7, True, self.input_sub_group_size, self.input_sub_group_mode)
                pass
            else:
                # calculate_outlier_exp(input, self.input_quant_grid, self.layer_id, self.layer_name, self.group_size, True)
                deq_input = input

                # distri_3d(deq_input, layer_idx=self.layer_id, layer_name=self.layer_name)

        # out = gemm_with_compensation_gpu(input, self.weight, q_group_size=self.group_size, quant_grid=self.input_quant_grid)

        out = F.linear(deq_input.to(self.weight.dtype), self.weight)
        assert torch.isnan(out).sum() == 0

        if self.print_stats:
            print(f"layer: {self.layer_id}, tensor: {self.layer_name}, a_bit_width: {self.a_bit}. group_size: {self.group_size}")

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    