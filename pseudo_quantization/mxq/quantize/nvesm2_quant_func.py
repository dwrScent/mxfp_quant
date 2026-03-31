import torch
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_MAX = 448.0

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


FP4_E2M1_GRID = torch.tensor(float_value(2, 1), device="cuda")
FP6_E2M3_GRID = torch.tensor(float_value(2, 3), device="cuda")
FP8_E5M3_GRID = torch.tensor(float_value(5, 3), device="cuda")
FP8_E4M4_GRID = torch.tensor(float_value(4, 4), device="cuda")
# Keep high-precision reciprocals here since this path still multiplies in floating point.
ratio_div_LUT = torch.tensor([1.0, 1.0 / 1.25, 1.0 / 1.5, 1.0 / 1.75])
err_LUT = torch.outer(
    torch.arange(17).pow(2),
    torch.tensor([4, 5, 6, 7]).pow(2),
)


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
    x_quant, indices = quantize_to_grid(x_abs, FP4_E2M1_GRID)
    return x_quant * sign
def cast_to_fp4_idx(x: torch.Tensor):
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    x_quant, indices = quantize_to_grid(x_abs, FP4_E2M1_GRID)
    return x_quant * sign, indices


# hardware friendly
@torch.no_grad()
def get_quant_nvesm2_hw(tensor_value: torch.Tensor, group_size: int):

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
    ratio = torch.tensor([1.0, 1.25, 1.5, 1.75], dtype=tensor_value.dtype, device=tensor_value.device)
    scales = scales.reshape(-1, 1).expand(-1, group_size // sub_group_size).reshape(-1, 1)
    x_expanded = tensor_value.unsqueeze(2)
    scales_expanded = scales.unsqueeze(2)
    cand_scales = scales_expanded * ratio.view(1, 1, -1)
    scales_expanded_div = 1 / scales_expanded
    cand_scales_div = scales_expanded_div * ratio_div_LUT.to(dtype=tensor_value.dtype, device=tensor_value.device).view(1, 1, -1)
    cand_qval = cast_to_fp4(x_expanded * cand_scales_div) * cand_scales

    norm_val = x_expanded * cand_scales_div
    err = (norm_val - cast_to_fp4(norm_val)).abs()
    err_code = torch.clamp(torch.round(err * 16), 0, 16).to(torch.long)
    ratio_idx = torch.arange(ratio.numel(), device=tensor_value.device, dtype=torch.long).view(1, 1, -1).expand_as(err_code)
    err_square = err_LUT.to(device=tensor_value.device)[err_code, ratio_idx]
    score_per_ratio = err_square.sum(dim=1)
    best_ratio_idx = score_per_ratio.argmin(dim=1)

    row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
    best_dqval = cand_qval[row_idx, :, best_ratio_idx]

    tensor_deq = best_dqval.reshape(org_shape).to(org_dtype)
    return tensor_deq
