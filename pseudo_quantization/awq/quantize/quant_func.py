
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
