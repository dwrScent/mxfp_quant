import torch

def rounding_error_mask(value_a: torch.float16, rounding_direction_mask=None):
    value_a_int = value_a.view(torch.int16)

    manti_a = value_a_int & 0x03FF  # 0x03FF = 0000001111111111
    exp_a = (value_a_int >> 10) & 0x1F  # 提取 5-bit exponent

    rounding_err_mask = torch.zeros_like(manti_a, dtype=torch.int8)

    # 情况 1：exponent == -1，即指数位为 01110

    exp_neg_1_mask = (exp_a == 0b01110)
    rouning_error_mask_1 = (((manti_a >> 6) & 0xf ) == 0x7)  # 判断 manti == 1.0111XXXXXX 的情况
    rounding_err_mask[exp_neg_1_mask & rouning_error_mask_1] = 1

    # 情况 2：exponent == -2，即指数位为 01101
    exp_neg_2_mask = (exp_a == 0b01101)
    rouning_error_mask_2 = (((manti_a >> 6) & 0xf ) == 0x8)  # 判断 manti == 1.1000XXXXXX 的情况
    rounding_err_mask[exp_neg_2_mask & rouning_error_mask_2] = 1  # 通常向上舍入（接近 0.5）

    # 情况 3：exponent <= -3，即指数位 <= 01100
    exp_neg_3_or_below_mask = (exp_a <= 0b01100)

    # 情况 4：exponent >= 0
    # .00XXXXXXXX 和 .10XXXXXXXX -> 向下舍入；.00111 和 .10111 认为是误差比较大的 case
    normal_exp_mask = (exp_a >= 0b01111)
    rounding_error_mask_3 = (((manti_a >> 5) & 0x1f ) == 0b00111) | (((manti_a >> 5) & 0x1f ) == 0b10111)
    rounding_err_mask[rounding_error_mask_3 & normal_exp_mask] = 1
    # .01XXXXXXXX 和 .11XXXXXXXX -> 向上舍入;.01000 和 .11000 认为是误差比较大的 case
    rounding_error_mask_4 = (((manti_a >> 5) & 0x1f ) == 0b01000) | (((manti_a >> 5) & 0x1f ) == 0b11000)
    rounding_err_mask[rounding_error_mask_4 & normal_exp_mask] = 1

    # for i in range(manti_a.shape[-1]):
    #     print(i, value_a[0][i], format(manti_a[0][i].item(), '#018b'), rounding_err_mask[0][i], rounding_direction_mask[0][i])
    # exit(0)

    return rounding_err_mask

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

    # NOTE: 有一个假定很重要，那就是对于 scale_e8m0，量化前后的 mantissa 不变（但其实也不是，还是会有 shift 变换；而且普通的 scale 量化后，再去统计 rounding 信息，也不会有特别大的挑战其实）
    tensor_value_int = quant_tensor.view(torch.int16)

    # 提取 mantissa (低 10 位) 和 exponent (第 10 到 14 位)
    mantissa = tensor_value_int & 0x03FF  # 0x03FF = 0000001111111111
    exponent = (tensor_value_int >> 10) & 0x1F  # 提取 5-bit exponent

    # 初始化 rounding_direction：-1 表示向下舍入，1 表示向上舍入
    rounding_direction = torch.zeros_like(mantissa, dtype=torch.int8)

    # 情况 1：exponent == -1，即指数位为 01110
    exp_neg_1_mask = (exponent == 0b01110)
    round_down_mask_1 = (((mantissa >> 9) & 0x01 ) == 0)  # 判断 .0XXXX 的情况
    round_up_mask_1 = (((mantissa >> 9) & 0x01 ) == 1)    # 判断 .1XXXX 的情况
    rounding_direction[exp_neg_1_mask & round_down_mask_1] = -1
    rounding_direction[exp_neg_1_mask & round_up_mask_1] = 1

    # 情况 2：exponent == -2，即指数位为 01101
    exp_neg_2_mask = (exponent == 0b01101)
    rounding_direction[exp_neg_2_mask] = 1  # 通常向上舍入（接近 0.5）

    # 情况 3：exponent <= -3，即指数位 <= 01100
    exp_neg_3_or_below_mask = (exponent <= 0b01100)
    rounding_direction[exp_neg_3_or_below_mask] = -1  # 通常向下舍入（接近 0）

    # 情况 4：exponent >= 0
    # .00XXXXXXXX 和 .10XXXXXXXX -> 向下舍入
    normal_exp_mask = (exponent >= 0b01111)
    round_down_mask = (((mantissa >> 8) & 0x01 ) == 0)
    rounding_direction[round_down_mask & normal_exp_mask] = -1
    # .01XXXXXXXX 和 .11XXXXXXXX -> 向上舍入
    round_up_mask = (((mantissa >> 8) & 0x01 ) == 1)
    rounding_direction[round_up_mask & normal_exp_mask] = 1
    # 如果所有 mantissa 刚好是 .0100000000 或者 .0000000000，则不舍入
    # BUG: corner case 1, 比如 1.25 * 2 = 2.50；按照 bit op 是向上舍入，但实际他是做了向下舍入
    # BUG: corner case 2, 对于 exponent 最大的元素，不管怎么样都是向下舍入
    no_round_mask = (mantissa == 0x000) | (mantissa == 0x100)
    rounding_direction[no_round_mask & normal_exp_mask] = 0

    # rounding_error_mask(quant_tensor, rounding_direction)

    # print(torch.sum(rounding_direction == 0))
    # exit(0)
    # labels = (((tensor_value + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # quant_tensor_round = quant_grid[labels] 

    # print(rounding_direction, rounding_direction.shape)
    # for i in range(tensor_value_int.shape[-1]):
    #     print(i, tensor_value[0][i], tensor_value_int[0][i], format(tensor_value_int[0][i].item(), '#018b'), rounding_direction[0][i])
    #     print(quant_tensor[0][i], quant_tensor_round[0][i], rounding_direction[0][i], '\n')

    # exit(0)

    return rounding_direction

def gemm_with_compensation(tensor_value: torch.float16, weight: torch.float16, q_group_size=32, quant_grid=None):
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

    quant_tensor = (tensor_value + zeros) / scales

    rounding_dir_mask = get_rounding_mask_bitwise(tensor_value, scales, q_group_size=q_group_size)
    rounding_err_mask = rounding_error_mask(quant_tensor)

    labels = (quant_tensor.unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    quantized_value = quant_grid[labels]
    tensor_deq = quantized_value * scales - zeros

    error_indices = (rounding_err_mask == 1).nonzero(as_tuple=True)
    print(error_indices[1], rounding_err_mask)
    print("Rounding Direction Mask:", rounding_dir_mask[error_indices])
    print("Quantized Value (quant_tensor):", quant_tensor[error_indices])
    print("Quantized Rounding Value (quantized_value):", quantized_value[error_indices])
    # exit(0)

    M, K, N = tensor_deq.shape[0], tensor_value.shape[1], weight.shape[1]
    output_tensor = torch.zeros((M, N), dtype=torch.float16)
    output_tensor_compensate = torch.zeros((M, N), dtype=torch.float16)

    tensor_value_int = quantized_value.view(torch.int16)
    exponent = (tensor_value_int >> 10) & 0x1F  # 提取 5-bit exponent
    exponent_tmp = torch.where(exponent - 15 >= 0, exponent - 15, 0) # 有 2 个 2^0 的 case
    exponent_value = torch.pow(2, exponent_tmp)  
    
    # print(exponent_value)

    # BUG: 最大值 rounding 方向的问题会影响这里的计算，要注意
    for i in range (M):
        for j in range (N):
            partial_sum = torch.tensor(0., dtype=torch.float32, device=tensor_deq.device)
            partial_sum_compensate = torch.tensor(0., dtype=torch.float32, device=tensor_deq.device)
            for k in range (K):
                compensate_value = torch.tensor(0., dtype=torch.float32, device=tensor_deq.device)
                if rounding_err_mask[i][k] == 1:
                    rounding_dir = rounding_dir_mask[i][k]
                    # 补偿 0.125，不那么激进
                    compensate_value = ((exponent_value[i][k] * 0.125) * rounding_dir * (-1))
                    if quantized_value[i][k] < 0:
                        compensate_value = - compensate_value
                    print(compensate_value)
                partial_sum += quantized_value[i][k] * weight[k][j]
                partial_sum_compensate += ((quantized_value[i][k]+compensate_value) * weight[k][j])
            output_tensor[i][j] = partial_sum * scales[i][0]
            output_tensor_compensate[i][j] = partial_sum_compensate * scales[i][0]
    # print('quantized output', output_tensor)
    # print('output with compensate', output_tensor_compensate)
    return output_tensor_compensate



# 示例测试
X = torch.randn(1, 32, dtype=torch.float16).cuda()
# X = torch.abs(torch.randn(1024, 32, dtype=torch.float16).cuda()) 
weight = torch.randn(32, 1, dtype=torch.float16).cuda()
quant_grid = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0], dtype=torch.float16).cuda()

# 调用 get_quant_mxfp 得到量化后的 tensor
tensor_deq, scales, rounding_mask_from_quant = get_quant_mxfp(X, quant_grid, q_group_size=32)


# 使用 bitwise 方法验证舍入方向
rounding_mask_bitwise = get_rounding_mask_bitwise(X, scales, q_group_size=32, quant_grid=quant_grid)

output_tensor_compensate = gemm_with_compensation(X, weight, q_group_size=32, quant_grid=quant_grid)

# 验证结果
matching_mask = (rounding_mask_from_quant == rounding_mask_bitwise)
num_mismatch = (~matching_mask).sum().item()
total_elements = rounding_mask_from_quant.numel()

print(f"Total elements: {total_elements}")
print(f"Number of mismatches in rounding direction: {num_mismatch}")
print(f"Mismatch percentage: {100 * num_mismatch / total_elements:.4f}%")

output_golden = torch.matmul(X, weight)
output_dequant = torch.matmul(tensor_deq, weight)

print('Init Output  ----->   ', output_golden)
print('Quantized Output  ----->   ', output_dequant)
print('Quantized Output with Compensation  ----->   ', output_tensor_compensate)