
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_func import get_quant_smxfp


@torch.no_grad()
def smxfp_search(tensor_value, quant_grid, q_group_size=-1):
    org_shape = tensor_value.shape
    tensor_value = tensor_value.reshape(-1, q_group_size)
    batch_num = 1
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
    val_info = torch.finfo(tensor_value.dtype)
    tensor_value = tensor_value.nan_to_num(0.0, val_info.min, val_info.max).clamp(val_info.min, val_info.max)
    assert torch.isinf(tensor_value).sum() == 0
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)
    
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0, f"Last dimension {org_shape[-1]} must be divisible by q_group_size {q_group_size}"
        tensor_value_reshaped = tensor_value.reshape(-1, q_group_size)
    else:
        tensor_value_reshaped = tensor_value.reshape(-1, org_shape[-1])
        
    num_q_groups = tensor_value_reshaped.shape[0]

    # Compute base exponent (same for all bias)
    max_val = tensor_value_reshaped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    max_quant_val = max(quant_grid)
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))

    # Prepare grids
    half = quant_grid.shape[0] // 2
    grid1_tensor = quant_grid[:half]
    grid2_tensor = quant_grid[half:]
    
    sub_group_size = 2
    if q_group_size > 0 and q_group_size % sub_group_size == 0:
        num_sub_groups_per_q_group = q_group_size // sub_group_size
    else:
        raise ValueError("q_group_size must be an even number (>0) for sub-grouping with fixed size 2.")

    # Handle outliers once (same across all bias trials)
    if keep_outlier:
        outlier_mask = torch.zeros_like(tensor_value_reshaped, dtype=torch.bool, device=tensor_value_reshaped.device)
        _, indices = torch.topk(tensor_value_reshaped.abs(), 1, dim=1)
        outlier_mask.scatter_(1, indices, 1)
        org_tensor_for_outlier = tensor_value_reshaped.clone()
        tensor_value_processed = tensor_value_reshaped * ~outlier_mask
    else:
        tensor_value_processed = tensor_value_reshaped.clone()

    # Reshape processed tensor for subgroup
    tensor_processed_subgrouped = tensor_value_processed.reshape(num_q_groups, num_sub_groups_per_q_group, sub_group_size)

    # Bias search
    bias_range = [-2, -1, 0, 1, 2]
    deq_candidates = []   # list of (num_q_groups, q_group_size)
    mse_candidates = []   # list of (num_q_groups, 1)

    for bias in bias_range:
        # Compute scale with bias
        scale = torch.pow(2.0, exp + bias)  # (num_q_groups, 1)
        assert not (scale == 0).any()

        # Normalize
        normalized = (tensor_processed_subgrouped) / scale.unsqueeze(-1)  # (N, S, 2)

        # Quantize with grid1
        labels1 = (normalized.unsqueeze(-1) - grid1_tensor).abs().argmin(dim=-1)
        dq1 = grid1_tensor[labels1] * scale.unsqueeze(-1)
        mse1 = (dq1 - tensor_processed_subgrouped).pow(2).mean(dim=-1)  # (N, S)

        # Quantize with grid2
        labels2 = (normalized.unsqueeze(-1) - grid2_tensor).abs().argmin(dim=-1)
        dq2 = grid2_tensor[labels2] * scale.unsqueeze(-1)
        mse2 = (dq2 - tensor_processed_subgrouped).pow(2).mean(dim=-1)

        # Choose per sub-group
        select_mask = (mse1 <= mse2).unsqueeze(-1)  # (N, S, 1)
        dq_subgrouped = torch.where(select_mask, dq1, dq2)  # (N, S, 2)

        # Reshape back to (num_q_groups, q_group_size)
        dq_group = dq_subgrouped.reshape(num_q_groups, q_group_size)

        # Apply outlier restoration
        if keep_outlier:
            dq_group = dq_group * ~outlier_mask + org_tensor_for_outlier * outlier_mask

        # Compute group-level MSE (mean over all elements in group)
        mse_group = (dq_group - tensor_value_reshaped).pow(2).mean(dim=1, keepdim=True)  # (N, 1)

        deq_candidates.append(dq_group)
        mse_candidates.append(mse_group)

    # Stack candidates
    all_deq = torch.stack(deq_candidates, dim=0)   # (3, N, q_group_size)
    all_mse = torch.stack(mse_candidates, dim=0)   # (3, N, 1)

    # Select best bias per group
    best_bias_idx = all_mse.argmin(dim=0).squeeze(-1)  # (N,)
    group_indices = torch.arange(num_q_groups, device=tensor_value.device)
    final_deq = all_deq[best_bias_idx, group_indices, :]  # (N, q_group_size)

    # Final sanity and reshape
    final_deq = final_deq.nan_to_num(0.0, val_info.min, val_info.max).clamp(val_info.min, val_info.max)
    assert torch.isinf(final_deq).sum() == 0
    assert torch.isnan(final_deq).sum() == 0

    tensor_deq = final_deq.reshape(org_shape)
    quant_mse_sum = all_mse[best_bias_idx, group_indices, :].reshape(num_q_groups, 1)

    quant_obj = 'input' if is_input else 'weight'
    if print_stats:
        print(f"Quantization MSE: {quant_mse_sum.mean().item()}, quant_obj: {quant_obj}, keep_outlier: {keep_outlier}")
    
    if get_labels:
        print("Warning: 'get_labels=True' is complex with sub-group logic. Returning dummy labels.")
        dummy_labels = torch.zeros(tensor_value_reshaped.shape, dtype=torch.long, device=tensor_value.device)
        return tensor_deq, quant_mse_sum, dummy_labels, exp.reshape(org_shape[0], 1)  # or scale? adjust as needed
    else:
        return tensor_deq, quant_mse_sum

class SMXFP_Linear(nn.Module):
    def __init__(self, w_bit, a_bit, group_size, in_features, out_features, bias, dev, ant_config, layer_id, layer_name):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.group_size = group_size 
        self.ant_config = ant_config

        self.layer_id = layer_id
        self.layer_name = layer_name

        # ANT param
        self.weight_quant_grid = None
        self.weight_alpha = -1
        self.input_quant_grid = None
        self.input_alpha = -1

        self.search_tag = None
        self.keep_outlier = False
        # self.keep_outlier = True

        self.print_stats = False
        # self.print_stats = True

        # SMXFP param
        self.weight_smxfp_mode = ant_config['weight_mxfp_mode']
        self.input_smxfp_mode = ant_config['input_mxfp_mode']

        self.weight_sub_group_size = ant_config.get('weight_sub_group_size')
        self.weight_sub_group_mode = ant_config.get('weight_sub_group_mode')
        self.input_sub_group_size = ant_config.get('input_sub_group_size')
        self.input_sub_group_mode = ant_config.get('input_sub_group_mode')

        assert self.in_features % self.group_size == 0

        self.exp_bit_width = None

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, a_bit, group_size, layer_id, layer_name, init_only=False, ant_config=None, quant_mode=None):

        smxfp_linear = cls(w_bit, a_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, ant_config, layer_id, layer_name)
        if init_only:  # just prepare for loading sd
            return smxfp_linear

        # smxfp_linear.weight = linear.weight.data.clone().half()
        smxfp_linear.weight = linear.weight.data
        
        if linear.bias is not None:
            smxfp_linear.bias = linear.bias.clone().half()
        
        fp4_e1m2 = torch.tensor([ 0.0000, -0.0000,  0.2500, -0.2500,  0.5000, -0.5000,  0.7500, -0.7500,
                                  0.0000, -0.0000,  0.5000, -0.5000,  1.0000, -1.0000,  1.5000, -1.5000,]
                                ,dtype=linear.weight.data.dtype)

        fp6_e1m4 = torch.tensor([ 0.0000, -0.0000,  0.0625, -0.0625,  0.1250, -0.1250,  0.1875, -0.1875, 
                                0.2500, -0.2500,  0.3125, -0.3125,  0.3750, -0.3750,  0.4375, -0.4375,
                                0.5000, -0.5000,  0.5625, -0.5625,  0.6250, -0.6250,  0.6875, -0.6875,
                                0.7500, -0.7500,  0.8125, -0.8125,  0.8750, -0.8750,  0.9375, -0.9375,
                                ]
                                ,dtype=linear.weight.data.dtype)
        fp6_e1m4 = torch.cat(fp6_e1m4, fp6_e1m4*2)
        if w_bit == 4:
            smxfp_linear.weight_quant_grid = fp4_e1m2
        elif w_bit == 6:
            smxfp_linear.weight_quant_grid = fp6_e1m4
        else:
            raise Exception(f"w_bit={w_bit} not support yet")
    
        if a_bit == 4:
            smxfp_linear.input_quant_grid = fp4_e1m2
        elif a_bit == 6:
            smxfp_linear.input_quant_grid = fp6_e1m4
        else:
            raise Exception(f"w_bit={w_bit} not support yet")

        # 参考论文 https://arxiv.org/pdf/2302.08007 里的配置，k1=16, k2=2
        assert smxfp_linear.group_size == 16
            
        return smxfp_linear
    
    def _quantize_data(self, data, mode, quant_grid, n_bit, exp_base, is_input, sub_group_size, sub_group_mode):
        # sub group with E0M3
        # sub_group_grid = [0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]     
        quantize_methods = {
            'base': lambda: get_quant_smxfp(data, quant_grid=quant_grid, q_group_size=self.group_size, is_input=is_input),
            'base_search': lambda: smxfp_search(data, quant_grid=quant_grid, q_group_size=self.group_size),
            # 'scale_search': lambda: smxfp_search(data, quant_grid=quant_grid, mode=None, zero_point=False, q_group_size=self.group_size, keep_outlier=self.keep_outlier, print_stats=self.print_stats),
            # 'dtype_search': lambda: dtype_search_v2(data, quant_grid=quant_grid, mode=None, zero_point=False, q_group_size=self.group_size, keep_outlier=self.keep_outlier),
            # 'dtype_search_olive': lambda: dtype_search_olive(data, quant_grid=quant_grid, mode=None, zero_point=False, q_group_size=self.group_size, n_bit=n_bit, exp_base=exp_base),
            # 'naive_adapt': lambda: smxfp_direct(data, quant_grid=quant_grid, mode=None, zero_point=False, q_group_size=self.group_size, n_bit=n_bit),
            # 'sub_group': lambda: smxfp_sub_group(data, quant_grid=quant_grid, sub_group_grid=sub_group_grid, mode=None, zero_point=False, q_group_size=self.group_size, sub_group_size=sub_group_size, sub_group_mode=sub_group_mode, print_stats=self.print_stats),
            # 'sub_group_adaptive': lambda: smxfp_sub_group_adaptive(data, quant_grid=quant_grid, sub_group_grid=sub_group_grid, mode=None, zero_point=False, q_group_size=self.group_size, sub_group_size=sub_group_size, print_stats=self.print_stats),
            # 'sub_group_v3': lambda: smxfp_sub_group_v3(data, quant_grid=quant_grid, sub_group_grid=sub_group_grid, mode=None, zero_point=False, q_group_size=self.group_size, print_stats=self.print_stats),
        }
        return quantize_methods.get(mode, lambda: NotImplementedError(f'not support this smxfp mode: {mode}'))()

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input = x.reshape(-1, x.shape[-1])

        # Search and set data type and alpha in the first inference
        if self.search_tag is None:
            # calculate_scale_range(self.weight, self.weight_quant_grid, self.layer_id, self.layer_name, self.group_size, False)
            # calculate_outlier_exp(self.weight, self.weight_quant_grid, self.layer_id, self.layer_name, self.group_size, False)

            # for weight
            # distri_3d(self.weight, layer_idx=self.layer_id, layer_name=self.layer_name)

            if self.w_bit < 16:
                deq_weight, _ = self._quantize_data(self.weight, self.weight_smxfp_mode, self.weight_quant_grid, self.w_bit, 5, False, self.weight_sub_group_size, self.weight_sub_group_mode)
            else:
                deq_weight = self.weight
            
            # Quantize weight only once
            self.weight = deq_weight

            if self.a_bit < 16:
                deq_input, _ = self._quantize_data(input, self.input_smxfp_mode, self.input_quant_grid, self.a_bit, 7, True, self.input_sub_group_size, self.input_sub_group_mode)
            else:
                deq_input = input
                
            self.search_tag = 1

        # quantize input based on the selected data type and alpha
        else:
            if self.a_bit < 16:
                # calculate_scale_range(input, self.input_quant_grid, self.layer_id, self.layer_name, self.group_size, True)

                deq_input, _ = self._quantize_data(input, self.input_smxfp_mode, self.input_quant_grid, self.a_bit, 7, True, self.input_sub_group_size, self.input_sub_group_mode)
            else:
                # calculate_outlier_exp(input, self.input_quant_grid, self.layer_id, self.layer_name, self.group_size, True)
                deq_input = input

                # distri_3d(deq_input, layer_idx=self.layer_id, layer_name=self.layer_name)

        # out = gemm_with_compensation_gpu(input, self.weight, q_group_size=self.group_size, quant_grid=self.input_quant_grid)

        out = F.linear(deq_input, self.weight)
        assert torch.isnan(out).sum() == 0

        if self.print_stats:
            print(f"layer: {self.layer_id}, tensor: {self.layer_name}, a_bit_width: {self.a_bit}. group_size: {self.group_size}")

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    