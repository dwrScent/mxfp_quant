import torch
LEVEL_2_MAX = 7.05


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
    # indices = torch.ones(indices.shape[0], 1).to(tensor_value.device, dtype=indices.dtype)
    outlier_mask = outlier_mask.reshape(-1, 4).scatter_(1, indices , 1)
    outlier_mask = outlier_mask.reshape(-1, group_size)
    in_grp = e1m2 * (1 - outlier_mask) + e1m4 * outlier_mask
    tensor_quant = sign * in_grp * (SF * 2.0 ** DE64)

    return tensor_quant.reshape(org_shape).to(org_dtype)


def get_quant_hifes(tensor_value: torch.Tensor, group_size: int):
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
    exp = torch.floor(torch.log2(v_max)) - torch.floor(torch.log2(torch.tensor(LEVEL_2_MAX, device=tensor_value.device)))
    man = torch.round(v_max / LEVEL_2_MAX / 2 ** (exp - 2)) * 2 ** (-2)
    range_ = range(-1, 2)
    bias_mse = {}
    for bias in range_:
        scales = torch.pow(2, exp + bias) * man
        ratio = [1.0, 1.25, 1.5, 1.75]
        scales_expanded = scales.unsqueeze(2)
        # cand_scale shape is (N_group, 1, 4)
        cand_scales = scales_expanded * torch.tensor(ratio, device=tensor_value.device, dtype=tensor_value.dtype).view(1, 1, -1)
        # x_expanded shape is (N_group, group_size, 1)
        x_expanded = tensor_value.abs().unsqueeze(2)
        E1_8_expanded = (v_max8.unsqueeze(2) / cand_scales) >= 4
        E1_8_expanded = E1_8_expanded.to(tensor_value.dtype)
        E1_8x2_expanded = E1_8_expanded.repeat_interleave(2, dim=1)
        E1_16_expanded = (v_max16.unsqueeze(2) / cand_scales * 2.0 ** (-E1_8x2_expanded)) >= 2
        E1_16_expanded = E1_16_expanded.to(tensor_value.dtype)
        DE16_expanded = E1_16_expanded + E1_8x2_expanded
        DE64_expanded = DE16_expanded.repeat_interleave(4, dim=1)
        cand_qval = torch.floor(x_expanded / cand_scales / 2 ** DE64_expanded * 4.0 + 0.5) * 2.0 ** (-2)
        cand_qval[cand_qval >= 2.0] = 1.75
        # print(sign.unsqueeze(2).shape, cand_qval.shape, cand_scales.shape, DE64_expanded.shape)
        cand_qval = cand_qval * cand_scales * 2 ** DE64_expanded
        mse_per_ratio = (cand_qval - x_expanded).pow(2).mean(dim=1)
        best_ratio_idx = mse_per_ratio.argmin(dim=1)
        row_idx = torch.arange(tensor_value.size(0), device=tensor_value.device)
        best_dqval = cand_qval[row_idx, :, best_ratio_idx]
        quant_mse_per_grp = mse_per_ratio[row_idx, best_ratio_idx]
        tensor_deq = best_dqval.reshape(-1, group_size)
        quant_mse_grp = quant_mse_per_grp.view(-1, 1)
        bias_mse[bias] = (tensor_deq, quant_mse_grp)
    all_mse = torch.cat([bias_mse[b][1] for b in range_], dim=1)
    best_bias_idx = all_mse.argmin(dim=1)
    all_deq = torch.stack([bias_mse[b][0] for b in range_], dim=0)
    all_deq = all_deq.view(len(range_), -1, group_size)
    idx_expanded = best_bias_idx.view(1, -1, 1).expand(1, -1, group_size)
    final_deq = torch.gather(all_deq, dim=0, index=idx_expanded).squeeze(0)
    tensor_deq = final_deq.reshape(org_shape).to(org_dtype) * sign.reshape(org_shape).to(org_dtype)
    return tensor_deq


def cast_to_E6M2(x: torch.Tensor):
    x = x.clamp(min=2 ** (-48) * 1.0, max=2 ** 15 * 1.5)
    E = torch.floor(torch.log2(x))
    return torch.round(x * 2 ** (-E + 2)) * 2 ** (E - 2)


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
    v_max8 = v_max16.reshape(tensor_value.shape[0], -1, 2).amax(dim=2)
    v_max = v_max8.amax(dim=1, keepdim=True)
    SF = cast_to_E6M2(v_max / LEVEL_2_MAX)
    E1_8 = (v_max8 / SF) >= 4
    E1_8 = E1_8.to(v_max8.dtype)
    E1_8x2 = E1_8.repeat_interleave(2, dim=1)
    E1_16 = (v_max16 / SF * 2.0 ** (-E1_8x2)) >= 2
    E1_16 = E1_16.to(v_max16.dtype)
    DE16 = E1_16 + E1_8x2
    DE64 = DE16.repeat_interleave(4, dim=1)
    # multiply 2^2, plus 0.5, floor, then multiply 2^-2, 
    # it's round to nearest with 2 bit mantissa
    in_grp = torch.floor(tensor_value.abs() / (SF * 2.0 ** (DE64 - 2)) + 0.5) * 2.0 ** (-2)
    in_grp[in_grp >= 2.0] = 1.75
    tensor_quant = sign * in_grp * (SF * 2.0 ** DE64)

    return tensor_quant.reshape(org_shape).to(org_dtype)


