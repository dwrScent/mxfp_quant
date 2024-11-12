import torch

def get_quant_mxfp(tensor_value, quant_grid, q_group_size=32):
    """
    Simplified quantization function with rounding direction tracking, considering the sign bit.
    Returns tensor_deq and rounding_mask.
    """
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    # Compute the scaling factor
    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    max_quant_val = max(quant_grid)
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp)
    zeros = 0

    # Perform quantization
    labels = (((tensor_value + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    tensor_deq = quant_grid[labels] * scales - zeros

    # Compute rounding mask, considering sign bit
    quant_tensor = (tensor_value + zeros) / scales
    quantized_value = quant_grid[labels]

    # Initialize rounding_mask: -1 for round down, 1 for round up
    rounding_mask = torch.zeros_like(quant_tensor, dtype=torch.int8)

    # Positive values
    positive_mask = tensor_value >= 0
    rounding_mask[positive_mask & (quant_tensor < quantized_value)] = 1   # Round up
    rounding_mask[positive_mask & (quant_tensor > quantized_value)] = -1  # Round down

    # Negative values
    negative_mask = tensor_value < 0
    rounding_mask[negative_mask & (quant_tensor > quantized_value)] = 1   # Round up
    rounding_mask[negative_mask & (quant_tensor < quantized_value)] = -1  # Round down

    tensor_deq = tensor_deq.reshape(org_shape)
    # rounding_mask = rounding_mask.reshape(org_shape)
    
    # print(torch.sum(rounding_mask == 0))
    print(rounding_mask)

    return tensor_deq, scales, rounding_mask

def get_rounding_mask_bitwise(tensor_value, scales, q_group_size=32, quant_grid=None):
    """
    Determine rounding direction for each element using bit operation.
    Returns a rounding mask with -1 for round down and 1 for round up.
    """
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    # 将 FP16 tensor 视为 int16 (short)
    zeros = 0
    quant_tensor = (tensor_value + zeros) / scales
    # print(quant_tensor, quant_tensor.shape)
    tensor_value_int = quant_tensor.view(torch.int16)

    # 提取 mantissa (低 10 位) 和 exponent (第 10 到 14 位)
    mantissa = tensor_value_int & 0x03FF  # 0x03FF = 0000001111111111
    exponent = (tensor_value_int >> 10) & 0x1F  # 提取 5-bit exponent

    # 初始化 rounding_direction：-1 表示向下舍入，1 表示向上舍入
    rounding_direction = torch.zeros_like(mantissa, dtype=torch.int8)

    # 判断舍入方向
    # .00XXXXXXXX 和 .10XXXXXXXX -> 向下舍入
    round_down_mask = (((mantissa >> 8) & 0x01 ) == 0)
    rounding_direction[round_down_mask] = -1

    # .01XXXXXXXX 和 .11XXXXXXXX -> 向上舍入
    round_up_mask = (((mantissa >> 8) & 0x01 ) == 1)
    rounding_direction[round_up_mask] = 1

    # 如果所有 mantissa 刚好是 .0100000000 或者 .0000000000，则不舍入
    no_round_mask = (mantissa == 0x000) | (mantissa == 0x200)
    rounding_direction[no_round_mask] = 0

    # print(torch.sum(rounding_direction == 0))
    # exit(0)
    labels = (((tensor_value + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    quant_tensor_round = quant_grid[labels] 

    print(rounding_direction, rounding_direction.shape)
    for i in range(tensor_value_int.shape[-1]):
        print(i, tensor_value[0][i], tensor_value_int[0][i], format(tensor_value_int[0][i].item(), '#018b'), mantissa[0][i], rounding_direction[0][i])
        print(quant_tensor[0][i], quant_tensor_round[0][i], rounding_direction[0][i], '\n')

    # exit(0)

    return rounding_direction

# 示例测试
# X = torch.randn(1024, 1024, dtype=torch.float16).cuda()
X = torch.abs(torch.randn(1, 32, dtype=torch.float16).cuda()) + 1.0
quant_grid = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0], dtype=torch.float16).cuda()

# 调用 get_quant_mxfp 得到量化后的 tensor
tensor_deq, scales, rounding_mask_from_quant = get_quant_mxfp(X, quant_grid, q_group_size=32)

# 使用 bitwise 方法验证舍入方向
rounding_mask_bitwise = get_rounding_mask_bitwise(X, scales, q_group_size=32, quant_grid=quant_grid)

# 验证结果
matching_mask = (rounding_mask_from_quant == rounding_mask_bitwise)
num_mismatch = (~matching_mask).sum().item()
total_elements = rounding_mask_from_quant.numel()

print(f"Total elements: {total_elements}")
print(f"Number of mismatches in rounding direction: {num_mismatch}")
print(f"Mismatch percentage: {100 * num_mismatch / total_elements:.4f}%")