
import torch
import torch.nn.functional as F
from torch import nn
from .ant_quant import get_quant_weight
from .utils_stats import calculate_max_error

def pseudo_quantize_int(tensor, n_bit=8, zero_point=False, q_group_size=-1, get_scale=False):
    org_shape = tensor.shape
    padding_size = 0
    
    if q_group_size > 0:
        if org_shape[-1] % q_group_size != 0:
            # Calculate padding size
            padding_size = q_group_size - (org_shape[-1] % q_group_size)
            # Apply padding
            tensor = F.pad(tensor, (0, padding_size), "constant", 0)
            padding_shape = tensor.shape
            assert padding_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    assert tensor.dim() == 2
    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)

        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(tensor).sum() == 0

    tensor = (torch.clamp(torch.round(tensor / scales) +
                        zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(tensor).sum() == 0

    if padding_size > 0:
        tensor = tensor.reshape(padding_shape)
        tensor = tensor[:, :org_shape[-1]]
    else:
        tensor = tensor.reshape(org_shape)

    if get_scale:
        return tensor, max_val, scales
    else:
        return tensor

def pseudo_quantize_giant(tensor, n_bit=8, zero_point=False, q_group_size=-1, get_scale=False):
    quantized_part_shape = tensor.shape

    if q_group_size > 0:
        quantized_part_group = tensor.reshape(-1, q_group_size)
    else:
        raise ValueError('not support yet')
    max_val = torch.max(torch.abs(quantized_part_group), dim=1, keepdim=True).values
    value_var = torch.var(quantized_part_group / max_val, dim=1, keepdim=True)

    quant_grid_set = {}
    quant_grid_set['coefficient_25'] = torch.tensor([-1.0000, -0.7061, -0.5181, -0.3828, -0.2739, -0.1782, -0.0891, -0.0033, 0.0033,  0.0891,  0.1782,  0.2739,  0.3828,  0.5181,  0.7061,  1.0000])
    quant_grid_set['int'] = torch.tensor([-0., -7., -6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5., 6.,  7.])
    quant_grid_set['coefficient_0'] = torch.tensor([-1.0000, -0.5000, -0.2500, -0.1250, -0.0625, -0.0312, -0.0156, -0.0078, 0.0078,  0.0156,  0.0312,  0.0625,  0.1250,  0.2500,  0.5000,  1.0000])


    quantized_part_group_deq_nf, _ = get_quant_weight(quantized_part_group, quant_grid_set['coefficient_25'], mode='coefficient_25', q_group_size=q_group_size)
    quantized_part_group_deq_int, _ = get_quant_weight(quantized_part_group, quant_grid_set['int'], mode='int', q_group_size=q_group_size)
    quantized_part_group_deq_pot, _ = get_quant_weight(quantized_part_group, quant_grid_set['coefficient_0'], mode='coefficient_0', q_group_size=q_group_size)

    mask_pot = (value_var < 0.05).expand_as(quantized_part_group_deq_pot)
    mask_nf = ((value_var >= 0.05) & (value_var <= 0.25)).expand_as(quantized_part_group_deq_nf)
    mask_int = (value_var > 0.25).expand_as(quantized_part_group_deq_int)

    # 使用 mask 选取对应的量化后 tensor
    quantized_part_group_deq = torch.zeros_like(quantized_part_group)

    quantized_part_group_deq = torch.where(mask_pot, quantized_part_group_deq_pot, quantized_part_group_deq)
    quantized_part_group_deq = torch.where(mask_nf, quantized_part_group_deq_nf, quantized_part_group_deq)
    quantized_part_group_deq = torch.where(mask_int, quantized_part_group_deq_int, quantized_part_group_deq)

    quantized_part_deq = quantized_part_group_deq.reshape(quantized_part_shape)
    quantized_part_deq = quantized_part_deq.to(dtype=tensor.dtype, device=tensor.device)

    assert torch.isnan(quantized_part_deq).sum() == 0

    return quantized_part_deq


@torch.no_grad()
def get_quant_grid(tensor_value, quant_grid, group_size, alpha=1.0):

    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)
    assert torch.isnan(tensor_value).sum() == 0


    max_quant_val = max(quant_grid)

    if group_size == -2:
        # tensor-wise
        tensor_value = tensor_value.view(-1)

        max_val = tensor_value.abs().amax()
        scales = (max_val * alpha) / max_quant_val
        zeros = 0

        batch_num = 4
        assert tensor_value.shape[0] % batch_num == 0
        batch_size = tensor_value.shape[0] // batch_num
        tensor_deq = torch.zeros_like(tensor_value)
        for idx in range(batch_num):
            tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size]
            labels = (((tensor_par + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
            tensor_q_par = quant_grid[labels] * scales - zeros
            tensor_deq[idx*batch_size : (idx+1)*batch_size] = tensor_q_par
        
    # channel or group wise
    elif group_size >= -1:

        if group_size > 0:
            assert org_shape[-1] % group_size == 0
            tensor_value = tensor_value.reshape(-1, group_size)

        assert tensor_value.dim() == 2

        max_val = tensor_value.abs().amax(dim=1, keepdim=True)
        scales = (max_val * alpha) / max_quant_val

        # if group_size > 0:
            # scales = scales.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float16)
            # scales = scales.to(dtype=torch.float8_e5m2).to(dtype=torch.float16)

        zeros = 0

        # Batch processing to avoid OOM
        # batch_num = 4
        batch_num = 4
        assert tensor_value.shape[0] % batch_num == 0
        batch_size = tensor_value.shape[0] // batch_num
        tensor_deq = torch.zeros_like(tensor_value)
        for idx in range(batch_num):
            tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
            labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
            tensor_q_par = quant_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
            tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par


    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(scales).sum() == 0

    # tensor_deq = tensor_deq.half()
    tensor_deq = tensor_deq.reshape(org_shape)

    # if group_size > 0:
    #     calculate_max_error(tensor_value, tensor_deq, q_group_size=group_size)

    return tensor_deq


@torch.no_grad()
def get_quant_mxfp(tensor_value, quant_grid, mode="int", zero_point=True, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False, is_input=False, keep_outlier=False, print_stats=False):
    '''
    return : dequantized weight, mse?
    '''
    assert torch.isinf(tensor_value).sum() == 0
    assert torch.isnan(tensor_value).sum() == 0

    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-5)

    max_quant_val = max(quant_grid)
    
    # Compute the scaling factor
    # pow(2, math.floor(math.log2(25)) - math.floor(math.log2(6)))
    # exp = torch.ceil(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    
    scales = (max_val * alpha) / max_quant_val

    # scales = scales.to(torch.float8_e3m4)


    assert torch.isinf(max_val).sum() == 0

    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp)

    assert not (scales == 0).any(), "Scale should contain 0 values"
    assert torch.isnan(scales).sum() == 0

    # exp_max_val = torch.floor(torch.log2(max_val))
    # mask = torch.where(tensor_value > torch.pow(2, exp_max_val), torch.tensor(1), torch.tensor(0))

    zeros = 0
    # org_value = tensor_value.clone()

    if keep_outlier:
        outlier_mask = torch.zeros_like(tensor_value, dtype=torch.bool).to(tensor_value.device)
        _, indices = torch.topk(tensor_value.abs(), 1)
        outlier_mask.scatter_(1, indices, 1)

        # print(outlier_mask.shape[0], "Original outlier_mask count of 1s:", outlier_mask.sum().item())
        org_tensor = tensor_value.clone()

        tensor_value = tensor_value * ~outlier_mask

    # Batch processing to avoid OOM
    # batch_num = 4
    batch_num = 4
    assert tensor_value.shape[0]  % batch_num == 0, \
    f"Batch dimension mismatch! Current tensor shape[0]={tensor_value.shape[0]},  batch_num={batch_num}. " \
    f"The first dimension of tensor ({tensor_value.shape[0]})  must be divisible by batch_num ({batch_num})"
    batch_size = tensor_value.shape[0] // batch_num
    tensor_deq = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        tensor_q_par = quant_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    # tensor_deq = org_value * mask + tensor_deq * (1-mask)
    quant_mse = (tensor_deq-tensor_value).abs().pow(2).to(torch.float32)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    # print('test', tensor_deq, scales, quant_mse, quant_mse_sum, quant_mse_sum.max())
    if keep_outlier:
        tensor_deq = tensor_deq * ~outlier_mask + org_tensor * outlier_mask


    assert torch.isinf(tensor_deq).sum() == 0
    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(scales).sum() == 0
    # assert torch.isnan(quant_mse).sum() == 0

    tensor_deq = tensor_deq.reshape(org_shape)

    # calculate_max_error(tensor_value, tensor_deq, q_group_size=q_group_size)

    quant_obj = 'input' if is_input else 'weight'
    if print_stats:
        print(f"Quantization MSE: {quant_mse_sum.mean().item()}, quant_obj: {quant_obj}, keep_outlier: {keep_outlier}")
    # print('init', scales, tensor_deq, tensor_value)

    if get_labels:
        return tensor_deq, quant_mse_sum, labels, quant_grid * scales
    else:
        return tensor_deq, quant_mse_sum
    

@torch.no_grad()
def get_quant_nvfp(tensor_value, quant_grid, mode="int", zero_point=True, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False, is_input=False, keep_outlier=False, print_stats=False):
    '''
    return : dequantized weight, mse?
    '''
    assert torch.isinf(tensor_value).sum() == 0
    assert torch.isnan(tensor_value).sum() == 0

    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-5)

    max_quant_val = max(quant_grid)
    assert torch.isinf(max_val).sum() == 0
    
    # Compute the scaling factor
    # pow(2, math.floor(math.log2(25)) - math.floor(math.log2(6)))
    # exp = torch.ceil(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    
    scales = (max_val * alpha) / max_quant_val
    scales = scales.clamp(min=0.001953125, max=448.0)

    # scale_fp16 = scales.clone()
    scales = scales.to(torch.float8_e4m3fn)
    # scales = scales.to(torch.float8_e4m3fnuz)
    scales = scales.to(torch.float16)

    # test_tensor = torch.tensor(272.5, device=scales.device, dtype=torch.float16)
    # print(test_tensor, test_tensor.to(torch.float8_e4m3fnuz), test_tensor.to(torch.float8_e4m3fn))
    # exit(0)

    # if not torch.isnan(scales).sum() == 0:
    #     nan_mask = torch.where(torch.isnan(scales), torch.tensor(1.0, device=scales.device), torch.tensor(0.0, device=scales.device))
    #     nan_max_val = max_val * nan_mask    
    #     nan_scales = scales * nan_mask
    #     nan_scales_fp16 = scale_fp16 * nan_mask

    #     print(max_val.max(), max_val.min(), scales, nan_max_val.max(), nan_max_val.min(), max_quant_val, nan_scales_fp16.max())
    #     exit(0)

    assert not (scales == 0).any(), "Scale should contain 0 values"
    assert torch.isnan(scales).sum() == 0

    # exp_max_val = torch.floor(torch.log2(max_val))
    # mask = torch.where(tensor_value > torch.pow(2, exp_max_val), torch.tensor(1), torch.tensor(0))

    zeros = 0
    # org_value = tensor_value.clone()

    if keep_outlier:
        outlier_mask = torch.zeros_like(tensor_value, dtype=torch.bool).to(tensor_value.device)
        _, indices = torch.topk(tensor_value.abs(), 1)
        outlier_mask.scatter_(1, indices, 1)


        # 对每 3 个 group，保留第 3 行（index % 3 == 2），其他两行清零
        # mask = torch.arange(outlier_mask.shape[0], device=outlier_mask.device) % 3 != 0
        # outlier_mask[mask] = 0

        # 使用 olive 的方法得到 victim 的位置
        # victim_odd = torch.roll(outlier_mask.view(-1), 1, -1)
        # victim_odd[::2] = 0
        # victim_even = torch.roll(outlier_mask.view(-1) & (~victim_odd), -1, -1)
        # victim_even[1::2] = 0
        # non_victim_mask = ~(victim_even | victim_odd)
        # non_victim_mask = non_victim_mask.reshape(tensor_value.shape)

        # print(outlier_mask.shape[0], "Original outlier_mask count of 1s:", outlier_mask.sum().item())
        org_tensor = tensor_value.clone()

        tensor_value = tensor_value * ~outlier_mask

    # Batch processing to avoid OOM
    # batch_num = 4
    batch_num = 4
    assert tensor_value.shape[0] % batch_num == 0
    batch_size = tensor_value.shape[0] // batch_num
    tensor_deq = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        tensor_q_par = quant_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    # tensor_deq = org_value * mask + tensor_deq * (1-mask)
    quant_mse = (tensor_deq-tensor_value).abs().pow(2).to(torch.float32)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    # print('test', tensor_deq, scales, quant_mse, quant_mse_sum, quant_mse_sum.max())
    if keep_outlier:
        tensor_deq = tensor_deq * ~outlier_mask + org_tensor * outlier_mask

    # if not torch.isnan(tensor_deq).sum() == 0:
    #     print(torch.isnan(tensor_deq).sum(), torch.isnan(scales).sum(), scales.max(), scales.min())
    #     exit(0)
    assert torch.isinf(tensor_deq).sum() == 0
    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(scales).sum() == 0
    # assert torch.isnan(quant_mse).sum() == 0

    tensor_deq = tensor_deq.reshape(org_shape)

    # calculate_max_error(tensor_value, tensor_deq, q_group_size=q_group_size)

    quant_obj = 'input' if is_input else 'weight'
    if print_stats:
        print(f"Quantization MSE: {quant_mse_sum.mean().item()}, quant_obj: {quant_obj}, keep_outlier: {keep_outlier}")
    # print('init', scales, tensor_deq, tensor_value)

    if get_labels:
        return tensor_deq, quant_mse_sum, labels, quant_grid * scales
    else:
        return tensor_deq, quant_mse_sum


@torch.no_grad()
def get_quant_smxfp(tensor_value, quant_grid, q_group_size=-1, is_input=False):
    org_shape = tensor_value.shape
    tensor_value = tensor_value.reshape(-1, q_group_size)
    batch_num = 1 if is_input else 4                      # 固定批次大小 
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
        deq_sub, _ = get_quant_smxfp_inner(
            sub_tensor,
            quant_grid,
            q_group_size=q_group_size,
            is_input=is_input
        )
        
        deq_list.append(deq_sub) 
    
    # concat 回完整张量 
    tensor_deq      = torch.cat(deq_list,  dim=0).reshape(org_shape)
    
    return tensor_deq, None

@torch.no_grad()
def get_quant_smxfp_inner(tensor_value, quant_grid, q_group_size=-1, get_labels=False, is_input=False, keep_outlier=False, print_stats=False):
    '''
    return : dequantized weight, mse?
    '''
    tensor_value = tensor_value.clamp(min=torch.finfo(torch.float16).min, max=torch.finfo(torch.float16).max)
    assert torch.isinf(tensor_value).sum() == 0
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape # Original shape, e.g., (128, 512)
    quant_grid = quant_grid.to(tensor_value.device)
    
    # Reshape for q_group_size processing
    # Example: if org_shape=(128, 512) and q_group_size=16, then tensor_value_reshaped will be (128*32, 16)
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0, f"Last dimension {org_shape[-1]} must be divisible by q_group_size {q_group_size}"
        tensor_value_reshaped = tensor_value.reshape(-1, q_group_size)
    else:
        # If q_group_size is not set, treat the whole last dim as one group (though this specific code
        # will error out below if sub_group_size is not -1, as it expects q_group_size to be even)
        tensor_value_reshaped = tensor_value.reshape(-1, org_shape[-1])
        
    num_q_groups = tensor_value_reshaped.shape[0] # Number of total q_groups

    # Calculate scales for each q_group
    max_val = tensor_value_reshaped.abs().amax(dim=1, keepdim=True) # Max value per q_group
    max_val = max_val.clamp(min=1e-5) # Avoid division by too small value
    max_quant_val = max(quant_grid)
    
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp) # scales shape: (num_q_groups, 1)
    
    assert not (scales == 0).any(), "Scale should contain 0 values"
    
    zeros = 0 # Assuming zero_point=False, otherwise integrate it here
    
    # Handle outliers (still at q_group level)
    if keep_outlier:
        outlier_mask = torch.zeros_like(tensor_value_reshaped, dtype=torch.bool, device=tensor_value_reshaped.device)
        # Using topk to find the largest absolute value in each q_group and mark it as outlier
        _, indices = torch.topk(tensor_value_reshaped.abs(), 1, dim=1) # indices shape: (num_q_groups, 1)
        outlier_mask.scatter_(1, indices, 1) # Mark the outlier position with True
        org_tensor_for_outlier = tensor_value_reshaped.clone() # Keep original values for outliers
        tensor_value_processed = tensor_value_reshaped * ~outlier_mask # Zero out outliers
    else:
        tensor_value_processed = tensor_value_reshaped.clone()

    # Define the two new grids for sub-group quantization
    half = quant_grid.shape[0]  // 2
    grid1_tensor = quant_grid[:half]
    grid2_tensor = quant_grid[half:]
    
    # Check if q_group_size is defined and is an even number for sub-grouping
    # This now effectively becomes the main path. If not met, it will raise an error.
    sub_group_size = 2
    if q_group_size > 0 and q_group_size % sub_group_size == 0:
        num_sub_groups_per_q_group = q_group_size // sub_group_size
    else:
        raise ValueError("q_group_size must be an even number (>0) for sub-grouping with fixed size 2.")
    
    # Reshape tensor_value_processed to expose sub-groups
    # From (num_q_groups, q_group_size) to (num_q_groups, num_sub_groups_per_q_group, sub_group_size)
    tensor_processed_subgrouped = tensor_value_processed.reshape(num_q_groups, num_sub_groups_per_q_group, sub_group_size)
    
    # Scale each q_group by its corresponding scale.
    # scales shape: (num_q_groups, 1, 1). This allows broadcasting with tensor_processed_subgrouped.
    # Example: (N, 1, 1) * (N, S, 2) -> (N, S, 2)
    normalized_sub_groups = (tensor_processed_subgrouped + zeros) / scales.unsqueeze(-1)

    # --- Quantize with grid1_tensor ---
    # Reshape normalized_sub_groups for element-wise comparison with grid1_tensor
    # From (num_q_groups, num_sub_groups_per_q_group, sub_group_size)
    # To (num_q_groups, num_sub_groups_per_q_group, sub_group_size, 1) for broadcasting with grid1_tensor
    
    # (N, S, 2, 1) - (4,) -> (N, S, 2, 4)
    # labels_grid1 shape: (num_q_groups, num_sub_groups_per_q_group, sub_group_size)
    labels_grid1 = ((normalized_sub_groups.unsqueeze(-1) - grid1_tensor).abs().argmin(dim=-1))
    
    # quantized_normalized_grid1 shape: (num_q_groups, num_sub_groups_per_q_group, sub_group_size)
    quantized_normalized_grid1 = grid1_tensor[labels_grid1]
    
    # dequantized_grid1 shape: (num_q_groups, num_sub_groups_per_q_group, sub_group_size)
    dequantized_grid1 = quantized_normalized_grid1 * scales.unsqueeze(-1) - zeros
    
    # Calculate MSE for grid1
    # MSE will be (num_q_groups, num_sub_groups_per_q_group), representing MSE for each sub-group
    mse_grid1 = (dequantized_grid1 - tensor_processed_subgrouped).abs().pow(2).mean(dim=-1)

    # --- Quantize with grid2_tensor ---
    labels_grid2 = ((normalized_sub_groups.unsqueeze(-1) - grid2_tensor).abs().argmin(dim=-1))
    quantized_normalized_grid2 = grid2_tensor[labels_grid2]
    dequantized_grid2 = quantized_normalized_grid2 * scales.unsqueeze(-1) - zeros
    mse_grid2 = (dequantized_grid2 - tensor_processed_subgrouped).abs().pow(2).mean(dim=-1)

    # Choose the grid that yields lower MSE for each sub-group
    # selection_mask shape: (num_q_groups, num_sub_groups_per_q_group)
    selection_mask = (mse_grid1 <= mse_grid2).unsqueeze(-1) # Add a new dim for broadcasting with data
    
    # tensor_deq_subgrouped shape: (num_q_groups, num_sub_groups_per_q_group, sub_group_size)
    tensor_deq_subgrouped = torch.where(selection_mask, dequantized_grid1, dequantized_grid2)

    # Flatten back to the original q_group_size shape
    tensor_deq = tensor_deq_subgrouped.reshape(num_q_groups, q_group_size)
    
    # Apply outlier handling if enabled
    if keep_outlier:
        tensor_deq = tensor_deq * ~outlier_mask + org_tensor_for_outlier * outlier_mask

    # Calculate final MSE (mean across elements in each q_group)
    # quant_mse shape: (num_q_groups, q_group_size)
    quant_mse = (tensor_deq - tensor_value_reshaped).abs().pow(2).to(torch.float32)
    # quant_mse_sum shape: (num_q_groups, 1)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)
    
    tensor_deq = tensor_deq.clamp(min=torch.finfo(torch.float16).min, max=torch.finfo(torch.float16).max)
    assert torch.isinf(tensor_deq).sum() == 0
    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(scales).sum() == 0
    
    # Reshape quantized tensor back to original input shape
    tensor_deq = tensor_deq.reshape(org_shape)
    
    quant_obj = 'input' if is_input else 'weight'
    if print_stats:
        print(f"Quantization MSE: {quant_mse_sum.mean().item()}, quant_obj: {quant_obj}, keep_outlier: {keep_outlier}")
    
    if get_labels:
        print("Warning: 'get_labels=True' is complex with sub-group logic. Returning dummy labels.")
        dummy_labels = torch.zeros(tensor_value_reshaped.shape, dtype=torch.long, device=tensor_value.device)
        return tensor_deq, quant_mse_sum, dummy_labels, scales.reshape(org_shape[0], 1)
    else:
        return tensor_deq, quant_mse_sum
