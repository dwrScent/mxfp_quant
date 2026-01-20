import torch
from hif4_quant_func import get_quant_hifes
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_EPS = 2 ** (-9)
FLOAT8_E4M4_EPS = 2 ** (-10)
FLOAT8_E4M3_MAX = 448.0
LEVEL_2_MAX = 7.05


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

def exp_man_value(exp_bit, man_bit):
    bias = -48
    values = []
    for i in range(2**exp_bit):
        for j in range(2**man_bit):
            values.append((2 ** (i - bias)) * (1 + j * 2 ** (-man_bit)))
    return values


# FP4_E2M1_GRID = torch.tensor(float_value(2, 1), device="cuda")
# FP6_E2M3_GRID = torch.tensor(float_value(2, 3), device="cuda")
FP4_E2M1_GRID = torch.tensor(float_value(2, 1))
FP6_E2M3_GRID = torch.tensor(float_value(2, 3))
FP8_E5M3_GRID = torch.tensor(float_value(5, 3))
FP8_E4M4_GRID = torch.tensor(float_value(4, 4))
E6M2_GRID = torch.tensor(exp_man_value(6, 2))

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
    # print("previous fp6:")
    # print(FP6_E2M3_GRID.to(x.device)[fp6_index])
    fp6_index.clamp_(min=fp4_index * 4 - 1, max=fp4_index * 4 + 2)
    fp6 = FP6_E2M3_GRID.to(x.device)[fp6_index]

    return fp4 * sign, fp6 * sign

def cast_to_E6M2(x: torch.Tensor):
    x = x.clamp(min=2 ** (-48) * 1.0, max=2 ** 15 * 1.5)
    E = torch.floor(torch.log2(x))
    return torch.round(x * 2 ** (-E + 2)) * 2 ** (E - 2)

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


FLOAT8_E5M3_MAX = 2 ** 16 * 1.75
@torch.no_grad()
def cast_to_E5M3(x: torch.Tensor):
    x_quant, _ = quantize_to_grid(x, FP8_E5M3_GRID)
    return x_quant
    # x = x.clamp(min=2 ** (-17), max=FLOAT8_E5M3_MAX)
    # E = torch.floor(torch.log2(x))
    # return torch.round(x * 2 ** (-E + 3)) * 2 ** (E - 3)
@torch.no_grad()
def get_quant_nvfpe5(tensor_value: torch.Tensor, group_size: int):

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()
    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    scales = max_val / FLOAT4_E2M1_MAX
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E5M3_MAX
    sign = torch.sign(scales)
    scales = cast_to_E5M3(scales.abs() / global_scale) * global_scale

    tensor_quant = cast_to_fp4(tensor_value / scales) * scales * sign

    return tensor_quant.reshape(org_shape).to(org_dtype)


FLOAT8_E4M4_MAX = 2 ** 8 * 1.875
@torch.no_grad()
def cast_to_E4M4(x: torch.Tensor):
    x_quant, _ = quantize_to_grid(x, FP8_E4M4_GRID)
    return x_quant


@torch.no_grad()
def get_quant_nvfpm4(tensor_value: torch.Tensor, group_size: int):

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()
    if group_size > 0:
        assert org_shape[-1] % group_size == 0
        tensor_value = tensor_value.reshape(-1, group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    scales = max_val / FLOAT4_E2M1_MAX
    # avoid divide a too small value
    global_scale = scales.max() / FLOAT8_E4M4_MAX
    sign = torch.sign(scales)
    scales = cast_to_E4M4((scales.abs() / global_scale).clamp(min=FLOAT8_E4M4_EPS)) * global_scale

    tensor_quant = cast_to_fp4(tensor_value / scales) * scales * sign

    return tensor_quant.reshape(org_shape).to(org_dtype)


@torch.no_grad()
def get_quant_nvfpm5(tensor_value: torch.Tensor, group_size: int):

    # sub_group_size = 4  # extra 2 bit for scale in subgroup
    # assert group_size % sub_group_size == 0

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
    global_scale = scales.max() / E4M5_MAX
    scales = cast_to_E4M5((scales / global_scale)) * global_scale
    tensor_quant = cast_to_fp4(tensor_value / scales) * scales
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
def get_quant_nvesem(tensor_value: torch.Tensor, group_size: int):

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
    # range_ = {0}
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
            [1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device
        )
        # ratios = torch.tensor(
        #     [1.0, 1.5], dtype=tensor_value.dtype, device=tensor_value.device
        # )
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

E4M5_MAX = 2 ** 8 * 1.9735
E4M5_GRID = torch.tensor(float_value(4, 5))
@torch.no_grad()
def cast_to_E4M5(x: torch.Tensor):
    x_quant, _ = quantize_to_grid(x, E4M5_GRID)
    return x_quant


@torch.no_grad()
def get_quant_nvem(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 4  # extra 2 bit for mantissa in subgroup
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
    outlier_mask.scatter_(1, indices, 1)
    outlier_group_mask = outlier_mask.reshape(-1, group_size)
    tensor_quant = (fp4 * (1 - outlier_group_mask) + fp6 * outlier_group_mask) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)


@torch.no_grad()
def get_quant_hif4(tensor_value: torch.Tensor, group_size: int):

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    assert group_size == 64
    tensor_value = tensor_value.reshape(-1, group_size)

    sign = torch.sign(tensor_value)

    v_max16 = torch.zeros((tensor_value.shape[0], 16), device=tensor_value.device)
    v_max8 = torch.zeros((tensor_value.shape[0], 8), device=tensor_value.device)
    v_max16 = tensor_value.abs().reshape(tensor_value.shape[0], -1, 4).amax(dim=2)
    v_max16 = v_max16.reshape(tensor_value.shape[0], 16)
    v_max8 = v_max16.reshape(tensor_value.shape[0], -1, 2).amax(dim=2)
    v_max8 = v_max8.reshape(tensor_value.shape[0], 8)
    v_max = v_max8.amax(dim=1, keepdim=True)
    SF = cast_to_E6M2(v_max / LEVEL_2_MAX)
    E1_8 = (v_max8 / SF) >= 4
    E1_8 = E1_8.to(v_max8.dtype)
    E1_8x2 = E1_8.repeat_interleave(2, dim=1)
    E1_16 = (v_max16 / SF * 2.0 ** (-E1_8x2)) >= 2
    E1_16 = E1_16.to(v_max16.dtype)
    DE16 = E1_16 + E1_8x2
    DE64 = DE16.repeat_interleave(4, dim=1)
    in_grp = torch.floor(tensor_value.abs() / (SF * 2.0 ** (DE64 - 2)) + 0.5) * 2.0 ** (-2)
    in_grp[in_grp >= 2.0] = 1.75
    tensor_quant = sign * in_grp * (SF * 2.0 ** DE64)

    return tensor_quant.reshape(org_shape).to(org_dtype)


@torch.no_grad()
def get_quant_hifem(tensor_value: torch.Tensor, group_size: int):

    org_shape = tensor_value.shape
    org_dtype = tensor_value.dtype

    tensor_value = tensor_value.float()

    assert group_size == 64
    tensor_value = tensor_value.reshape(-1, group_size)

    sign = torch.sign(tensor_value)

    v_max16 = torch.zeros((tensor_value.shape[0], 16), device=tensor_value.device)
    v_max8 = torch.zeros((tensor_value.shape[0], 8), device=tensor_value.device)
    v_max16, indices = tensor_value.abs().reshape(tensor_value.shape[0], -1, 4).max(dim=2)
    # v_max16 = v_max16.reshape(tensor_value.shape[0], 16)
    v_max8 = v_max16.reshape(tensor_value.shape[0], -1, 2).amax(dim=2)
    # v_max8 = v_max8.reshape(tensor_value.shape[0], 8)
    v_max = v_max8.amax(dim=1, keepdim=True)
    SF = cast_to_E6M2(v_max / LEVEL_2_MAX)
    E1_8 = (v_max8 / SF) >= 4
    E1_8 = E1_8.to(v_max8.dtype)
    E1_8x2 = E1_8.repeat_interleave(2, dim=1)
    E1_16 = (v_max16 / SF * 2.0 ** (-E1_8x2)) >= 2
    E1_16 = E1_16.to(v_max16.dtype)
    DE16 = E1_16 + E1_8x2
    DE64 = DE16.repeat_interleave(4, dim=1)
    in_grp = tensor_value.abs() / (SF * 2.0 ** (DE64)) 
    e1m2 = torch.floor(in_grp * 2.0 ** 2 + 0.5) * 2.0 ** (-2)
    e1m4 = torch.floor(in_grp * 2.0 ** 4 + 0.5) * 2.0 ** (-4)
    outlier_mask = torch.zeros_like(tensor_value, dtype=tensor_value.dtype).to(
        tensor_value.device
    )
    e1m2[e1m2 >= 2.0] = 1.75
    e1m4[e1m4 >= 2.0] = 1.9375
    indices = indices.view(-1, 1)
    outlier_mask = outlier_mask.reshape(-1, 4).scatter_(1, indices , 1)
    outlier_mask.scatter_(1, indices, 1)
    outlier_mask = outlier_mask.reshape(-1, group_size)
    in_grp = e1m2 * (1 - outlier_mask) + e1m4 * outlier_mask
    tensor_quant = sign * in_grp * (SF * 2.0 ** DE64)

    return tensor_quant.reshape(org_shape).to(org_dtype)


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


__name__ = "__main__"

# a = torch.tensor([-0.27, 10.26, 6.41, 10.78, 9.25, 45.36, 10.72, 1.26])
# a = torch.tensor([-0.27, 10.26, 6.41, 70.08, 9.25, 45.36, 10.72, 1.26])
# a = a.repeat(2, 8)
#
b = torch.randn(128) * 100
print(b)
res = get_quant_hif4(b, 64)
mse = (res - b).pow(2).mean()

print(res)
print("MSE HIF4:", mse)

# res = get_quant_nvfpm4(b, 16)
# mse5 = (res - b).pow(2).mean()
# print(res)
# print("MSE NVFPM4:", mse5)
#
# res = get_quant_nvfpm5(b, 16)
# mse6 = (res - b).pow(2).mean()
# print(res)
# print("MSE NVFPM5:", mse6)

res = get_quant_nvfp(b, 16)
mse3 = (res - b).pow(2).mean()
print(res)
print("MSE NVFP:", mse3)

res = get_quant_nves(b, 16)
mse4 = (res - b).pow(2).mean()
print(res)
print("MSE NVES:", mse4)

res = get_quant_nvint4(b, 16)
mse6 = (res - b).pow(2).mean()
print(res)
print("MSE NVINT4:", mse6)

# res = get_quant_nvess(b, 16)
# mse5 = (res - b).pow(2).mean()
# print(res)
# print("MSE NVESS:", mse5)
#




import torch
import matplotlib.pyplot as plt

# 存储结果用于绘图
x_axis = []

mse1, mse2, mse3, mse4 = [], [], [], []
mse5, mse6, mse7, mse8 = [], [], [], []

# 采样次数
num_samples = 100

# import numpy as np
# # print distribution of x_val in one graph
# for j in range(1, 34):
#     if j % 4 != 0:
#         continue
#     x_val = j / 2
#     sigma = 0.01 * 2 ** (j / 2)
#     x_axis.append(x_val)
#     samples = torch.randn(10000) * sigma
#     # 1. 使用 histogram 统计频数
#     # density=True 可以让 Y 轴变为密度，方便不同方差下的对比
#     counts, bin_edges = np.histogram(samples.numpy(), bins=100, density=True)
#     # 2. 计算 bin 的中心点作为 X 轴坐标
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     # 3. 绘制折线图
#     plt.plot(bin_centers, counts, label=f'σ={sigma:.2f}', alpha=0.8)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Distribution of Random Samples for Different Variances')
# plt.legend()
# plt.grid(True, which="both", ls="-", alpha=0.5)
# # plt.savefig('distribution_of_samples.png')
# plt.show()
from scipy.stats import norm
import numpy as np

for j in range(1, 12):
    if j % 2 != 0:
        continue
    sigma = 0.01 * 2 ** (j / 2)
    
    # 生成 X 轴范围（根据当前 sigma 的 4 倍标准差设定）
    x = np.linspace(-4 * sigma, 4 * sigma, 200)
    
    # 计算理论正态分布的 PDF 值
    y = norm.pdf(x, 0, sigma)
    
    plt.plot(x, y, label=f'σ={sigma:.2f}', linewidth=2)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Random Samples for Different Variances')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('distribution_of_samples.png')
plt.show()
for j in range(1, 34):
    x_val = j / 2
    x_axis.append(x_val) # 修正1：填充横坐标
    sigma = 0.01 * 2 ** (j / 2)
    m1, m2, m3, m4 = 0.0, 0.0, 0.0, 0.0
    m5, m6, m7, m8 = 0.0, 0.0, 0.0, 0.0
    # 计算当前信号的理论方差，用于归一化
    # 因为 b = randn * (sigma^2)，其方差是 (sigma^2)^2
    signal_variance = sigma ** 2

    for i in range(num_samples):
        b = torch.randn(8192) * sigma 
        # 假设这些函数已经在你的命名空间中定义
        res1 = get_quant_hif4(b, 64)
        res2 = get_quant_nvfp(b, 16)
        res3 = get_quant_nves(b, 16)
        res4 = get_quant_nvem(b, 16)
        res5 = get_quant_mxfp(b, 32)
        res6 = get_quant_mxem(b, 32)
        res7 = get_quant_mxes(b, 32)
        res8 = get_quant_nvesem2(b, 16)

        # 累加 MSE
        m1 += (res1 - b).pow(2).mean().item()
        m2 += (res2 - b).pow(2).mean().item()
        m3 += (res3 - b).pow(2).mean().item()
        m4 += (res4 - b).pow(2).mean().item()
        m5 += (res5 - b).pow(2).mean().item()
        m6 += (res6 - b).pow(2).mean().item()
        m7 += (res7 - b).pow(2).mean().item()
        m8 += (res8 - b).pow(2).mean().item()

    # 修正2：均值除以样本数，再除以信号方差实现归一化
    mse1.append((m1 / num_samples) / signal_variance)
    mse2.append((m2 / num_samples) / signal_variance)
    mse3.append((m3 / num_samples) / signal_variance)
    mse4.append((m4 / num_samples) / signal_variance)
    mse5.append((m5 / num_samples) / signal_variance)
    mse6.append((m6 / num_samples) / signal_variance)
    mse7.append((m7 / num_samples) / signal_variance)
    mse8.append((m8 / num_samples) / signal_variance)

# --- 绘图部分 ---
plt.figure(figsize=(12, 8))

# 使用线性坐标轴，因为已经归一化了，数值应该在可比范围内
plt.plot(x_axis, mse1, label='HIF4 (4.5)', marker='o', markersize=4)
plt.plot(x_axis, mse2, label='NVFP (4.5)', marker='s', markersize=4)
plt.plot(x_axis, mse3, label='NVES (4.75)', marker='^', markersize=4)
plt.plot(x_axis, mse4, label='NVEM (4.75)', marker='x', markersize=4)
plt.plot(x_axis, mse5, label='MXFP (4.25)', marker='D', markersize=6)
plt.plot(x_axis, mse6, label='MXEM (4.5)', marker='v', markersize=4)
plt.plot(x_axis, mse7, label='MXES (4.5)', marker='*', markersize=4)
plt.plot(x_axis, mse8, label='NVESEM2 (4.625)', marker='P', markersize=4)


plt.xlabel('x variance = 0.01*2^(x)')
plt.ylabel('Normalized MSE (MSE / Variance)')
plt.title('Normalized Quantization Error vs. Signal Range')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# 保存并显示
plt.savefig('quant_mse_comparison.png')
plt.show()
