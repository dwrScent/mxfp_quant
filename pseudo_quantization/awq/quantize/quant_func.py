
import torch
import torch.nn.functional as F
from torch import nn
from .ant_quant import get_quant_weight

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

        batch_num = 32
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
        zeros = 0

        # Batch processing to avoid OOM
        batch_num = 1
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

    return tensor_deq

@torch.no_grad()
def get_quant_mxfp(tensor_value, quant_grid, mode="int", zero_point=True, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False):
    '''
    return : dequantized weight, mse?
    '''
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    zero_point = False

    if zero_point:
        # assert 0, "Not support zero point in ant quant now."
        max_val = tensor_value.amax(dim=1, keepdim=True)
        min_val = tensor_value.amin(dim=1, keepdim=True)
            # max_quant_val = max(abs(quant_grid)) * 2
        max_quant_val = max(quant_grid)
        min_quant_val = min(quant_grid)
        exp = torch.floor(torch.log2((max_val - min_val).clamp(min=1e-5))) / torch.floor(torch.log2(max_quant_val - min_quant_val))
        scales = torch.pow(2, exp)
        # fp16 z.p.
        # zeros = (- (max_quant_val + min_quant_val)) / 2  
        zeros = (- (max_val + min_val)) / 2

    else:
        max_val = tensor_value.abs().amax(dim=1, keepdim=True)

        if pos_value is None or pos_value == True:
            max_quant_val = max(quant_grid)
        elif pos_value == False:
            max_quant_val = abs(min(quant_grid))
        else:
            raise NotImplementedError 
        
        # Compute the scaling factor
        # pow(2, math.floor(math.log2(25)) - math.floor(math.log2(6)))
        exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
        # exp = torch.ceil(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
        # scales = (max_val * alpha) / max_quant_val
        scales = torch.pow(2, exp)

        # exp_max_val = torch.floor(torch.log2(max_val))
        # mask = torch.where(tensor_value > torch.pow(2, exp_max_val), torch.tensor(1), torch.tensor(0))

        zeros = 0

    # org_value = tensor_value.clone()
    keep_outlier = True
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
        # tensor_value = tensor_value * non_victim_mask


    # Batch processing to avoid OOM
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


    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(quant_mse).sum() == 0

    tensor_deq = tensor_deq.reshape(org_shape)

    if get_labels:
        return tensor_deq, quant_mse_sum, labels, quant_grid * scales
    else:
        return tensor_deq, quant_mse_sum