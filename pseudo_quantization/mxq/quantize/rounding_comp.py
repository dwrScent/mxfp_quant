import torch
import torch.nn.functional as F

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

def get_quant_mxfp(tensor_value, quant_grid, q_group_size=32, print_stat=False):
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
    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isinf(tensor_deq).sum() == 0
    
    # print(torch.sum(rounding_mask == 0))
    if print_stat:
        print(rounding_mask)

    return tensor_deq, scales, rounding_mask

def get_rounding_mask_bitwise(tensor_value, scales, q_group_size=32, quant_grid=None, print_stat=False):
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

    # quant_tensor_max = quant_tensor.abs().amax(dim=1, keepdim=True)
    # quant_tensor_max_int = quant_tensor_max.view(torch.int16)
    # exponent_max = (quant_tensor_max_int >> 10) & 0x1F

    # 提取 mantissa (低 10 位) 和 exponent (第 10 到 14 位)
    mantissa = tensor_value_int & 0x03FF  # 0x03FF = 0000001111111111
    exponent = (tensor_value_int >> 10) & 0x1F  # 提取 5-bit exponent

    # 初始化 rounding_direction：-1 表示向下舍入，1 表示向上舍入
    rounding_direction = torch.zeros_like(mantissa, dtype=torch.int8)

    # Case 1：exponent == -1，即指数位为 01110
    exp_neg_1_mask = (exponent == 0b01110)
    # 用 0x01 因为高位还有 sign, exp 的信息，屏蔽掉
    round_down_mask_1 = (((mantissa >> 9) & 0x01 ) == 0)  # 判断 .0XXXX 的情况
    round_up_mask_1 = (((mantissa >> 9) & 0x01 ) == 1)    # 判断 .1XXXX 的情况
    assert (round_down_mask_1 & round_up_mask_1).sum() == 0
    rounding_direction[exp_neg_1_mask & round_down_mask_1] = -1
    rounding_direction[exp_neg_1_mask & round_up_mask_1] = 1

    # Case 2：exponent == -2，即指数位为 01101
    exp_neg_2_mask = (exponent == 0b01101)
    rounding_direction[exp_neg_2_mask] = 1  # 通常向上舍入（接近 0.5）

    # Case 3：exponent <= -3，即指数位 <= 01100
    exp_neg_3_or_below_mask = (exponent <= 0b01100)
    rounding_direction[exp_neg_3_or_below_mask] = -1  # 通常向下舍入（接近 0）

    # Case 4：exponent >= 0
    # .00XXXXXXXX 和 .10XXXXXXXX -> 向下舍入
    normal_exp_mask = ((exponent >= 0b01111) & (exponent < 0b10001))
    round_down_mask_2 = (((mantissa >> 8) & 0x01 ) == 0)
    rounding_direction[round_down_mask_2 & normal_exp_mask] = -1
    # .01XXXXXXXX 和 .11XXXXXXXX -> 向上舍入
    round_up_mask_2 = (((mantissa >> 8) & 0x01 ) == 1)
    rounding_direction[round_up_mask_2 & normal_exp_mask] = 1

    assert (round_down_mask_2 & round_up_mask_2).sum() == 0
    # 如果所有 mantissa 刚好是 .0100000000 或者 .0000000000，则不舍入
    # BUG: corner case 1, 比如 1.25 * 2 = 2.50；按照 bit op 是向上舍入，但实际他是做了向下舍入
    no_round_mask = (mantissa == 0x000) | (mantissa == 0x100)
    rounding_direction[no_round_mask & normal_exp_mask] = 0

    # Case 5：exponent == 2，最大值的 exponent 一定是 0b10001；还可能有其他的较大值
    # 这个 case 下，只有 .01XXXXXXXX 是向上舍入，而 .00, .10, .11 都是向下舍入，主要是 .11 之上没有格点了
    max_exp_mask = (exponent == 0b10001)
    round_down_mask_3 = (((mantissa >> 8) & 0x03 ) != 0b01)
    rounding_direction[round_down_mask_3 & max_exp_mask] = -1
    round_up_mask_3 = (((mantissa >> 8) & 0x03 ) == 0b01)
    rounding_direction[round_up_mask_3 & max_exp_mask] = -1
    assert (round_down_mask_3 & round_up_mask_3).sum() == 0
    assert (exp_neg_1_mask & exp_neg_2_mask & exp_neg_3_or_below_mask & normal_exp_mask & max_exp_mask).sum() == 0
    # print(format(mantissa[0][3].item(), '#012b'), round_up_mask_3)
    

    # rounding_error_mask(quant_tensor, rounding_direction)

    # print(torch.sum(rounding_direction == 0))
    # exit(0)
    # labels = (((tensor_value + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # quant_tensor_round = quant_grid[labels] 

    # print(rounding_direction, rounding_direction.shape)
    # for i in range(tensor_value_int.shape[-1]):
    #     # 打印的字符串包含前缀 0b，占了 2 位，所以是 #018b，前缀 2 位加二进制 16 位
    #     print(i, tensor_value[0][i], tensor_value_int[0][i], format(tensor_value_int[0][i].item(), '#018b'), rounding_direction[0][i])
    #     print(quant_tensor[0][i], quant_tensor_round[0][i], rounding_direction[0][i], '\n')

    # exit(0)

    return rounding_direction.reshape(org_shape)

def gemm_with_compensation_gpu(tensor_value: torch.float16, weight: torch.float16, q_group_size=32, quant_grid=None, print_stat=False):
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device, dtype=torch.float16)

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    keep_outlier = False
    if keep_outlier:
        outlier_mask = torch.zeros_like(tensor_value, dtype=torch.bool).to(tensor_value.device)
        _, indices = torch.topk(tensor_value.abs(), 1)
        outlier_mask.scatter_(1, indices, 1)
        org_tensor = tensor_value.clone()
        tensor_value = tensor_value * ~outlier_mask

    # Compute the scaling factor
    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    max_quant_val = max(quant_grid)
    # exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    # scales = torch.pow(2, exp).to(torch.float16)

    scales = (max_val/ max_quant_val).to(torch.float16)

    zeros = 0

    # print('tensor_value, scales', tensor_value.shape, scales.shape)

    quant_tensor = (tensor_value + zeros) / scales

    rounding_dir_mask = get_rounding_mask_bitwise(tensor_value, scales, q_group_size=q_group_size)
    rounding_err_mask = rounding_error_mask(quant_tensor)

    labels = (quant_tensor.unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    quantized_value = quant_grid[labels]
    tensor_deq = quantized_value * scales - zeros

    if keep_outlier:
        tensor_deq = tensor_deq * ~outlier_mask + org_tensor * outlier_mask

    if print_stat:
        error_indices = (rounding_err_mask == 1).nonzero(as_tuple=True)
        print(error_indices[1], rounding_err_mask)
        print("Rounding Direction Mask:", rounding_dir_mask[error_indices])
        print("Quantized Value (quant_tensor):", quant_tensor[error_indices])
        print("Quantized Rounding Value (quantized_value):", quantized_value[error_indices])
    # exit(0)

    # print('quantized_value, tensor_deq', quantized_value.shape, tensor_deq.shape, quantized_value.dtype)

    tensor_value_int = quantized_value.view(torch.int16)
    # print('tensor_value_int', tensor_value_int.shape, quantized_value.shape)
    exponent = (tensor_value_int >> 10) & 0x1F  # 提取 5-bit exponent
    exponent_tmp = torch.where(exponent - 15 >= 0, exponent - 15, 0) # 有 2 个 2^0 的 case
    exponent_value = torch.pow(2, exponent_tmp)  
    # print('tensor_value_int', tensor_value_int.shape, exponent_tmp.shape, exponent.shape)

    negtive_mask = ((tensor_value_int >> 15) & 0x1 )
    pos_neg_mask = torch.where(negtive_mask == 1, -1, 1)

    # print(negtive_mask, tensor_value_int, quantized_value)
    # print(exponent_value.shape, rounding_dir_mask.shape, rounding_err_mask.shape, pos_neg_mask.shape)
    compensate_value = ((exponent_value * 0.125 * rounding_dir_mask * rounding_err_mask * pos_neg_mask) * (-1)).to(torch.float16) # compensate 要和原本的误差值相反
    # print('compensate gpu', exponent_value, compensate_value, rounding_err_mask, compensate_value.shape, scales.shape, scales.dtype)
    # print(compensate_value.shape, scales.shape)
    assert compensate_value.shape[0] == scales.shape[0]
    compensate_value_deq = compensate_value * scales

    # Reshape
    tensor_deq = tensor_deq.reshape(org_shape)
    compensate_value_deq = compensate_value_deq.reshape(org_shape)

    # print('ours', scales, tensor_deq, tensor_value)

    # output_deq = torch.matmul(tensor_deq, weight)
    output_deq = F.linear(tensor_deq, weight)
    # output_compensate = torch.matmul(compensate_value_deq, weight)
    output_compensate = F.linear(compensate_value_deq, weight)

    # print(output_deq, output_deq.shape, output_compensate, output_compensate.shape)

    return (output_deq + output_compensate)
    # return output_deq
