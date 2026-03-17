
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
