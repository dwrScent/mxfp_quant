import torch
import torch.nn.functional as F
from .hif4_quant_func import get_quant_hifes, get_quant_hifem, get_quant_hif4
from .nvesm2_quant_func import get_quant_nvesm2_hw
from torch import nn

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
# FLOAT8_E4M3_EPS = 2 ** (-9)
FLOAT8_E4M4_EPS = 2 ** (-10)
FLOAT8_E4M3_MAX = 448.0
LEVEL_2_MAX = 7


@torch.no_grad()
def fp16(tensor_value: torch.Tensor, group_size: int):
    return tensor_value


def float_value(exp_bit, man_bit):
    bias = (2 ** (exp_bit - 1)) - 1
    values = []
    min_to_zero = True
    subnormal = True
    for i in range(2**exp_bit):
        for j in range(2**man_bit):
            if min_to_zero:
                values.append(0.0)
                min_to_zero = False
            else:
                if subnormal:
                    values.append((2 ** (1 - bias)) * (j * 2 ** (-man_bit)))
                else:
                    values.append((2 ** (i - bias)) * (1 + j * 2 ** (-man_bit)))

        subnormal = False

    return values
# def float_value(exp_bit, man_bit):
#     bias = 0
#     values = []
#     min_to_zero = True
#     subnormal = True
#     for i in range(2**exp_bit):
#         for j in range(2**man_bit):
#             if min_to_zero:
#                 values.append(0.0)
#                 min_to_zero = False
#             else:
#                 if subnormal:
#                     values.append((2 ** (i - bias)) * (j * 2 ** (-man_bit)))
#                 else:
#                     values.append((2 ** (i - 1 - bias)) * (1 + j * 2 ** (-man_bit)))
#
#         subnormal = False
#
#     return values
#

FP4_E2M1_GRID = torch.tensor(float_value(2, 1), device="cuda")
FP6_E2M3_GRID = torch.tensor(float_value(2, 3), device="cuda")
FP8_E5M3_GRID = torch.tensor(float_value(5, 3), device="cuda")
FP8_E4M4_GRID = torch.tensor(float_value(4, 4), device="cuda")


def quantize_to_grid(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    levels = levels.to(x.device)
    boundaries = (levels[:-1] + levels[1:]) / 2.0
    odd_boundaries = boundaries[1::2]
    mask = torch.isin(x, odd_boundaries)
    x = x + 0.0000005 * mask  # round to even
    indices = torch.bucketize(x, boundaries)
    indices.clamp_(0, len(levels) - 1)

    quantized = levels[indices]
    return quantized, indices


def cast_to_fp4(x: torch.Tensor):
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    x_quant, _ = quantize_to_grid(x_abs, FP4_E2M1_GRID)
    return x_quant * sign


def cast_to_fp4_em(x: torch.Tensor):
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    fp4, fp4_index = quantize_to_grid(x_abs, FP4_E2M1_GRID)
    _, fp6_index = quantize_to_grid(x_abs, FP6_E2M3_GRID)
    fp6_index.clamp_(min=fp4_index * 4 - 1, max=fp4_index * 4 + 2)
    fp6 = FP6_E2M3_GRID.to(x.device)[fp6_index]

    return fp4 * sign, fp6 * sign


@torch.no_grad()
def get_quant_mxfp(tensor_value: torch.Tensor, group_size: int):

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()
    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    # Compute the scaling factor
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp)
    tensor_quant = cast_to_fp4(tensor_value / scales) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)


@torch.no_grad()
def get_quant_nvfp(tensor_value: torch.Tensor, group_size: int):

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()
    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    scales = max_val / FLOAT4_E2M1_MAX
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    tensor_quant = cast_to_fp4(tensor_value / scales) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)


@torch.no_grad()
def get_quant_mxes(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # Compute the scaling factor
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    bias_mse = {}
    range_ = range(-1, 2)
    for bias in range_:
        scales = torch.pow(2, exp + bias)
        sub_groups_per_group = group_size // sub_group_size
        # turn scales to (N_subgroups, 1)
        scales = scales.expand(-1, sub_groups_per_group).reshape(-1, 1)
        ratios = torch.tensor(
            [1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device
        )
        x_expanded = tensor_value.unsqueeze(2)
        scales_expanded = scales.unsqueeze(2)

        cand_scales = scales_expanded * ratios.view(1, 1, -1)
        cand_qval = cast_to_fp4(x_expanded / cand_scales) * cand_scales
        mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
        best_ratio_idx = mse_per_ratio.argmin(dim=1)
        row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
        best_dqval = cand_qval[row_idx, :, best_ratio_idx]
        quant_mse_per_subgrp = mse_per_ratio[row_idx, best_ratio_idx]
        tensor_deq = best_dqval.reshape(-1, group_size)
        quant_mse_sum = quant_mse_per_subgrp.view(-1, sub_groups_per_group).mean(
            dim=1, keepdim=True
        )
        bias_mse[bias] = (tensor_deq, quant_mse_sum)
    all_mse = torch.cat([bias_mse[b][1] for b in range_], dim=1)
    best_bias_idx = all_mse.argmin(dim=1)
    all_deq = torch.stack([bias_mse[b][0] for b in range_], dim=0)
    all_deq = all_deq.view(len(range_), -1, group_size)
    idx_expanded = best_bias_idx.view(1, -1, 1).expand(1, -1, group_size)
    final_deq = torch.gather(all_deq, dim=0, index=idx_expanded).squeeze(0)
    tensor_deq = final_deq.reshape(org_shape).to(org_dtype)
    return tensor_deq


@torch.no_grad()
def get_quant_mxem(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for mantissa in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    # Compute the scaling factor
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp)

    fp4, fp6 = cast_to_fp4_em(tensor_value / scales)

    tmp = fp4.reshape(-1, sub_group_size)
    outlier_mask = torch.zeros_like(tmp, dtype=tensor_value.dtype).to(
        tensor_value.device
    )

    _, indices = torch.topk(tmp.abs(), 1)
    outlier_mask.scatter_(1, indices, 1)
    outlier_group_mask = outlier_mask.reshape(-1, group_size)
    tensor_quant = (fp4 * (1 - outlier_group_mask) + fp6 * outlier_group_mask) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)


@torch.no_grad()
def get_quant_nves(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    )
    # print("org_scales:", scales)
    exp = torch.floor(torch.log2(scales))
    man_value = scales / torch.pow(2, exp)
    bias_mse = {}
    range_ = range(-1, 2)
    for bias in range_:
        # scales = torch.pow(2, exp + bias)

        scales = torch.pow(2, exp) * torch.pow(2, torch.tensor(bias, device=tensor_value.device, dtype=tensor_value.dtype))
        sub_groups_per_group = group_size // sub_group_size
        # N_subgroups, 1
        scales = scales.expand(-1, sub_groups_per_group).reshape(-1, 1)
        ratios = torch.tensor(
            [0, 0.03125, 0.0625, 0.09375], dtype=tensor_value.dtype, device=tensor_value.device
        )
        x_expanded = tensor_value.unsqueeze(2)
        scales_expanded = scales.unsqueeze(2)
        man_value_expanded = man_value.expand(-1, sub_groups_per_group).reshape(-1, 1).unsqueeze(2)

        cand_scales = scales_expanded * ratios.view(1, 1, -1) + scales_expanded * man_value_expanded
        # print(cand_scales, scales_expanded * man_value_expanded)
        cand_qval = cast_to_fp4(x_expanded / cand_scales / global_scale) * cand_scales * global_scale
        mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
        best_ratio_idx = mse_per_ratio.argmin(dim=1)
        row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
        best_dqval = cand_qval[row_idx, :, best_ratio_idx]
        quant_mse_per_subgrp = mse_per_ratio[row_idx, best_ratio_idx]
        tensor_deq = best_dqval.reshape(-1, group_size)
        quant_mse_sum = quant_mse_per_subgrp.view(-1, sub_groups_per_group).mean(
            dim=1, keepdim=True
        )
        bias_mse[bias] = (tensor_deq, quant_mse_sum)
    all_mse = torch.cat([bias_mse[b][1] for b in range_], dim=1)
    best_bias_idx = all_mse.argmin(dim=1)
    all_deq = torch.stack([bias_mse[b][0] for b in range_], dim=0)
    all_deq = all_deq.view(len(range_), -1, group_size)
    idx_expanded = best_bias_idx.view(1, -1, 1).expand(1, -1, group_size)
    final_deq = torch.gather(all_deq, dim=0, index=idx_expanded).squeeze(0)
    tensor_deq = final_deq.reshape(org_shape).to(org_dtype)
    return tensor_deq


@torch.no_grad()
def get_quant_nvesm(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale
    exp = torch.floor(torch.log2(scales))
    # exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    bias_mse = {}
    # range_ = range(-1, 2)
    range_ = {0}
    org_scales = scales
    for bias in range_:
        # scales = torch.pow(2, exp + bias)

        scales = org_scales * torch.pow(2, torch.tensor(bias, device=tensor_value.device, dtype=tensor_value.dtype))
        sub_groups_per_group = group_size // sub_group_size
        scales = scales.expand(-1, sub_groups_per_group).reshape(-1, 1)
        ratios = torch.tensor(
            [1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device
        )
        # ratios = torch.tensor(
        #     [1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device
        # )
        x_expanded = tensor_value.unsqueeze(2)
        scales_expanded = scales.unsqueeze(2)

        cand_scales = scales_expanded * ratios.view(1, 1, -1)
        cand_qval = cast_to_fp4(x_expanded / cand_scales) * cand_scales
        # mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
        mse_per_ratio = (cand_qval - x_expanded).abs().mean(dim=1)
        best_ratio_idx = mse_per_ratio.argmin(dim=1)
        row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
        best_dqval = cand_qval[row_idx, :, best_ratio_idx]
        quant_mse_per_subgrp = mse_per_ratio[row_idx, best_ratio_idx]
        tensor_deq = best_dqval.reshape(-1, group_size)
        quant_mse_sum = quant_mse_per_subgrp.view(-1, sub_groups_per_group).mean(
            dim=1, keepdim=True
        )
        bias_mse[bias] = (tensor_deq, quant_mse_sum)
    all_mse = torch.cat([bias_mse[b][1] for b in range_], dim=1)
    best_bias_idx = all_mse.argmin(dim=1)
    all_deq = torch.stack([bias_mse[b][0] for b in range_], dim=0)
    all_deq = all_deq.view(len(range_), -1, group_size)
    idx_expanded = best_bias_idx.view(1, -1, 1).expand(1, -1, group_size)
    final_deq = torch.gather(all_deq, dim=0, index=idx_expanded).squeeze(0)
    tensor_deq = final_deq.reshape(org_shape).to(org_dtype)
    return tensor_deq
@torch.no_grad()
def get_quant_nvesm2(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # ratio = torch.tensor([1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device)
    ratio = torch.tensor([1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device)
    scales = scales.reshape(-1, 1).expand(-1, group_size // sub_group_size).reshape(-1, 1)
    x_expanded = tensor_value.unsqueeze(2)
    scales_expanded = scales.unsqueeze(2)
    cand_scales = scales_expanded * ratio.view(1, 1, -1)
    cand_qval = cast_to_fp4(x_expanded / cand_scales) * cand_scales
    mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
    # mse_per_ratio = (cand_qval - x_expanded).abs().mean(dim=1)
    best_ratio_idx = mse_per_ratio.argmin(dim=1)
    row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
    best_dqval = cand_qval[row_idx, :, best_ratio_idx]

    tensor_deq = best_dqval.reshape(org_shape).to(org_dtype)
    return tensor_deq


@torch.no_grad()
def get_quant_nvesem2(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val

    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale
    exp = torch.floor(torch.log2(scales))
    # exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    bias_mse = {}
    range_ = range(-1, 2)
    org_scales = scales
    for bias in range_:
        # scales = torch.pow(2, exp + bias)

        scales = org_scales * torch.pow(2, torch.tensor(bias, device=tensor_value.device, dtype=tensor_value.dtype))
        sub_groups_per_group = group_size // sub_group_size
        scales = scales.expand(-1, sub_groups_per_group).reshape(-1, 1)
        ratios = torch.tensor(
            [1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device
        )
        x_expanded = tensor_value.unsqueeze(2)
        scales_expanded = scales.unsqueeze(2)

        cand_scales = scales_expanded * ratios.view(1, 1, -1)
        cand_qval = cast_to_fp4(x_expanded / cand_scales) * cand_scales
        mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
        best_ratio_idx = mse_per_ratio.argmin(dim=1)
        row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
        best_dqval = cand_qval[row_idx, :, best_ratio_idx]
        quant_mse_per_subgrp = mse_per_ratio[row_idx, best_ratio_idx]
        tensor_deq = best_dqval.reshape(-1, group_size)
        quant_mse_sum = quant_mse_per_subgrp.view(-1, sub_groups_per_group).mean(
            dim=1, keepdim=True
        )
        bias_mse[bias] = (tensor_deq, quant_mse_sum)
    all_mse = torch.cat([bias_mse[b][1] for b in range_], dim=1)
    best_bias_idx = all_mse.argmin(dim=1)
    all_deq = torch.stack([bias_mse[b][0] for b in range_], dim=0)
    all_deq = all_deq.view(len(range_), -1, group_size)
    idx_expanded = best_bias_idx.view(1, -1, 1).expand(1, -1, group_size)
    final_deq = torch.gather(all_deq, dim=0, index=idx_expanded).squeeze(0)
    tensor_deq = final_deq.reshape(org_shape).to(org_dtype)
    return tensor_deq


@torch.no_grad()
def get_quant_nvem(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 8  # extra 2 bit for mantissa in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    scales = max_val / FLOAT4_E2M1_MAX
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    fp4, fp6 = cast_to_fp4_em(tensor_value / scales)
    # print(fp4, "\n", fp6)

    tmp = fp4.reshape(-1, sub_group_size)
    outlier_mask = torch.zeros_like(tmp, dtype=tensor_value.dtype).to(
        tensor_value.device
    )

    _, indices = torch.topk(tmp.abs(), 1)
    # indices = torch.ones_like(indices) # experiment for em1
    outlier_mask.scatter_(1, indices, 1)
    outlier_group_mask = outlier_mask.reshape(-1, group_size)
    tensor_quant = (fp4 * (1 - outlier_group_mask) + fp6 * outlier_group_mask) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)


@torch.no_grad()
def get_quant_nvint4(tensor_value: torch.Tensor, group_size: int):
    
    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    scales = max_val / 7.0
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    tensor_quant = torch.clamp(torch.round(tensor_value / scales), min=-7.0, max=7.0) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)


@torch.no_grad()
def get_quant_nvintesm2(tensor_value: torch.Tensor, group_size=16):

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    scales = max_val / 7.0
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    sub_group_size = 8
    tensor_value = tensor_value.reshape(-1, sub_group_size)
    # ratio = torch.tensor([1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device)
    ratio = torch.tensor([1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device)
    scales = scales.reshape(-1, 1).expand(-1, group_size // sub_group_size).reshape(-1, 1)
    x_expanded = tensor_value.unsqueeze(2)
    scales_expanded = scales.unsqueeze(2)
    cand_scales = scales_expanded * ratio.view(1, 1, -1)
    cand_qval = torch.clamp(torch.round(x_expanded / cand_scales), min=-7.0, max=7.0) * cand_scales
    mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
    best_ratio_idx = mse_per_ratio.argmin(dim=1)
    row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
    best_dqval = cand_qval[row_idx, :, best_ratio_idx]

    tensor_quant = best_dqval

    return tensor_quant.reshape(org_shape).to(org_dtype)


def get_quant_gt4(tensor_value: torch.Tensor, group_size):
    sub_group_size = 8  # extra 2 bit for scale in subgroup
    assert group_size % sub_group_size == 0

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()
    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-8)

    max_quant_val = torch.tensor(FLOAT4_E2M1_MAX, device=tensor_value.device)

    scales = tensor_value.abs().amax(dim=1, keepdim=True) / max_quant_val

    # Compute the scaling factor
    global_scale = scales.max() / FLOAT8_E4M3_MAX
    scales = (
        (scales / global_scale)
        .clamp(min=FLOAT8_E4M3_EPS)
        .to(torch.float8_e4m3fn)
        .to(tensor_value.dtype)
    ) * global_scale

    subgroup_per_group = group_size // sub_group_size
    scales = scales.expand(tensor_value.shape[0], subgroup_per_group).reshape(-1).unsqueeze(1)
    tensor_value = tensor_value.reshape(-1, sub_group_size)

    num_gt_2 = (tensor_value.abs() / scales > 4).sum(dim=1, keepdim=True) / sub_group_size
    extra_scale = 1 + 0.5 * ( num_gt_2 >= 0.25 )
    # num_gt_2 = (tensor_value.abs() / scales > 2).sum(dim=1, keepdim=True)
    # extra_scale = 1 + 0.5 * (num_gt_2 >= 4).float()

    tensor_quant = cast_to_fp4(tensor_value / (scales * extra_scale)) * (scales * extra_scale)

    return tensor_quant.reshape(org_shape).to(org_dtype)


QUANT_METHOD_MAP = {
    "mxfp": get_quant_mxfp,
    "nvfp": get_quant_nvfp,
    "mxes": get_quant_mxes,
    "mxem": get_quant_mxem,
    "nves": get_quant_nves,
    "nvem": get_quant_nvem,
    "nvesm": get_quant_nvesm,
    "nvesm2": get_quant_nvesm2,
    "nvesm2_hw": get_quant_nvesm2_hw,
    "nvesem2": get_quant_nvesem2,
    "nvint4": get_quant_nvint4,
    "nvintesm2": get_quant_nvintesm2,
    "hif4": get_quant_hif4,
    "hifem": get_quant_hifem,
    "hifes": get_quant_hifes,
    "nvgt4": get_quant_gt4,
}


class QuantUnit:
    def __init__(self, bit: int, mode: str, group_size: int):
        self.bit = bit
        self.mode = mode
        self.group_size = group_size

        if self.bit == 16:
            self.quant_func = fp16
        elif self.bit == 4:
            assert mode in QUANT_METHOD_MAP
            self.quant_func = QUANT_METHOD_MAP[mode]

            if "nv" in mode:
                assert self.group_size == 16
            elif "hif" in mode:
                assert self.group_size == 64
            else:
                assert self.group_size == 32
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.quant_func(x, self.group_size)


class QuantConfig:
    def __init__(
        self, w_bit: int, w_mode: str, a_bit: int, a_mode: str, group_size: int
    ):
        self.w_unit = QuantUnit(w_bit, w_mode, group_size)
        self.a_unit = QuantUnit(a_bit, a_mode, group_size)

    def weight(self, weight):
        return self.w_unit.forward(weight)

    def activation(self, acitivation):
        return self.a_unit.forward(acitivation)
