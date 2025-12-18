import torch

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_MAX = 448.0


@torch.no_grad()
def fp16(tensor_value: torch.Tensor, group_size: int):
    return tensor_value


def float_value(exp_bit, man_bit):
    bias = 0
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
                    values.append((2 ** (i - bias)) * (j * 2 ** (-man_bit)))
                else:
                    values.append((2 ** (i - 1 - bias)) * (1 + j * 2 ** (-man_bit)))

        subnormal = False

    return values


# FP4_E2M1_GRID = torch.tensor(float_value(2, 1), device="cuda")
# FP6_E2M3_GRID = torch.tensor(float_value(2, 3), device="cuda")
FP4_E2M1_GRID = torch.tensor(float_value(2, 1))
FP6_E2M3_GRID = torch.tensor(float_value(2, 3))


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
    print("previous fp6:")
    print(FP6_E2M3_GRID.to(x.device)[fp6_index])
    fp6_index.clamp_(min=fp4_index * 4 - 1, max=fp4_index * 4 + 2)
    fp6 = FP6_E2M3_GRID.to(x.device)[fp6_index]

    return fp4 * sign, fp6 * sign


def get_quant_mxem(tensor_value: torch.Tensor, group_size: int):

    sub_group_size = 4  # extra 2 bit for mantissa in subgroup
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
    print(fp4, "\n", fp6)

    tmp = fp4.reshape(-1, sub_group_size)
    outlier_mask = torch.zeros_like(tmp, dtype=tensor_value.dtype).to(
        tensor_value.device
    )

    _, indices = torch.topk(tmp.abs(), 1)
    outlier_mask.scatter_(1, indices, 1)
    outlier_group_mask = outlier_mask.reshape(-1, group_size)
    tensor_quant = (fp4 * (1 - outlier_group_mask) + fp6 * outlier_group_mask) * scales

    return tensor_quant.reshape(org_shape).to(org_dtype)


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
    print(fp4, "\n", fp6)

    tmp = fp4.reshape(-1, sub_group_size)
    outlier_mask = torch.zeros_like(tmp, dtype=tensor_value.dtype).to(
        tensor_value.device
    )

    _, indices = torch.topk(tmp.abs(), 1)
    outlier_mask.scatter_(1, indices, 1)
    outlier_group_mask = outlier_mask.reshape(-1, group_size)
    tensor_quant = (fp4 * (1 - outlier_group_mask) + fp6 * outlier_group_mask) * scales
    print("scales:", scales.reshape(-1))

    return tensor_quant.reshape(org_shape).to(org_dtype)

__name__ = "__main__"

a = torch.tensor([-0.27, 10.26, 6.41, 10.78, 9.25, 45.36, 10.72, 1.26])

res = get_quant_nvem(a, 8)

print(res)
