
import torch
import torch.nn as nn
from .ant_quant import generate_quant_grid
import torch.nn.functional as F

def encode_gen(w_bit, return_list=False):
    """
    Generate the computable codebook from the index list
    Computable codebook: a*index + 2^index * b
    """
    coefficient_list = []
    # 选定系数 a
    for coefficient in range(0, 128, 10):
        coefficient_list.append(coefficient)
    # supply some specific data type, merge them after removing duplicates
    supply_list = [0, 5, 17, 20]
    merged_list = list(set(coefficient_list + supply_list))

    codebook_dict = {}
    b = 1
    for coefficient in merged_list:
    # for coefficient in coefficient_list:
        codebook_list = []
        # for item in range((2 ** w_bit) // 2 - 1):
        for item in range(2 ** w_bit):
            # 0~15 -> -7~8
            index = item - ((2 ** (w_bit-1)) - 1)
            if index < 0:
                index = (-index)
                codebook_list.append(-(coefficient * index + (2 ** index * b)))
            elif index == 0:
                codebook_list.append(0.)
            elif index > 0 and index < (2 ** w_bit // 2):
                codebook_list.append(coefficient * index + (2 ** index * b))
            elif index == (2 ** w_bit // 2):
                codebook_list.append(-0.)

        codebook_list = torch.tensor(codebook_list).to(dtype=torch.half)
        codebook_list.sort()
        
        # Normalization
        codebook_list = codebook_list / codebook_list.max()
        # to list if need
        if return_list:
            codebook_list = codebook_list.tolist()

        codebook_dict[f"coefficient_{coefficient}"] = codebook_list

    return codebook_dict

# core quantization method (simulated quantization)
def pseudo_quantize_int(tensor, n_bit=8, zero_point=True, q_group_size=-1):
    org_shape = tensor.shape
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
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
    tensor = tensor.reshape(org_shape)
    return tensor
    
@torch.no_grad()
def get_quant(tensor_value, quant_grid, alpha=1.0, is_input=False):

    quant_grid = quant_grid.to(tensor_value.device)

    if is_input:
        max_val = tensor_value.abs().amax()
    else:
        max_val = tensor_value.abs().amax(dim=1, keepdim=True)

    max_quant_val = max(quant_grid)
    scales = (max_val * alpha) / max_quant_val
    zeros = 0

    labels = (((tensor_value + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    tensor_deq = quant_grid[labels] * scales - zeros

    # quant_mse = (tensor_deq - tensor_value).abs().pow(2)
    tensor_deq = tensor_deq.to(tensor_value.device).half()

    return tensor_deq

def codeant_quant(self, n_bit, weight, input, ant_config, group_size, layer_id, layer_name, is_input=False):
    # channel-wise quantization for weight, tensor-wise for activation
    assert group_size == -1, "ANT use per-channel quantization for weight"
    # mode_list = ant_config['ant_mode'].split('-')
    mode_list = []
    # quant_grid_set = generate_quant_grid(n_bit=n_bit, signed=True, ant_mode=ant_config['ant_mode'])
    quant_grid_set = encode_gen(n_bit)
    mode_list.extend(quant_grid_set.keys())

    if group_size == -1:
        group_size = weight.shape[1]
    group_num = weight.shape[1] // group_size

    tensor_stats = {}
    for mode in mode_list:
        tensor_stats[mode] = torch.tensor(0.)


    deq_w = torch.zeros_like(weight, dtype=torch.half).to(weight.device)

    final_tensor = torch.zeros_like(weight, dtype=torch.half).to(weight.device)

    for group_id in range(0, group_num):
        x = input[ : , group_id * group_size: (group_id + 1) * group_size ] 
        org_group_w = weight[ : ,group_id * group_size: (group_id + 1) * group_size ]  
        org_group_output = torch.mm(x, org_group_w.T)

        deq_w = torch.zeros_like(org_group_w, dtype=torch.half).to(weight.device)
        min_mse = torch.full([1, weight.shape[0]], 1e5).to(weight.device) # 1 x N
        data_type_identify = torch.zeros_like(min_mse, dtype=torch.int32)
        mapping_list = {}
        for idx, mode in enumerate(mode_list):
            quant_grid = quant_grid_set[mode]

            w_group_deq = get_quant(weight, quant_grid, alpha=1.0, is_input=False)
            deq_group_output = torch.mm(x, w_group_deq.T).to(torch.float)

            print(w_group_deq, quant_grid)
            exit(0)

            mse = (deq_group_output - org_group_output).pow(2).mean(dim=0, keepdim=True) # M x N -> 1 x N

            sig = (mse <= min_mse).to(torch.half) # 1 x N
            mask = sig.repeat(group_size, 1).T # N x group_size
            org_mask = 1.0 - mask
            deq_w = torch.mul(deq_w, org_mask) + torch.mul(w_group_deq, mask)

            mapping_list[mode] = idx
            data_type_identify = torch.where(mse < min_mse, idx, data_type_identify)

            # update min MSE
            min_mse = torch.where(mse <= min_mse, mse, min_mse)

        data_type_mask = {}
        for mode in mode_list:
            data_type_mask[mode] = (data_type_identify == mapping_list[mode])
            tensor_stats[mode] = tensor_stats[mode].to(data_type_identify.device)
            tensor_stats[mode] = tensor_stats[mode] + torch.count_nonzero(data_type_identify.view(-1) == mapping_list[mode]) 
        final_tensor[ : ,group_id * group_size: (group_id + 1) * group_size ] = deq_w

    return final_tensor
class CODEANT_Linear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, ant_config, layer_id, layer_name):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size 
        self.ant_config = ant_config

        self.layer_id = layer_id
        self.layer_name = layer_name

        # CODE-ANT param
        self.weight_coefficient_a = None
        self.weight_quant_grid = None
        self.weight_alpha = -1
        self.input_quant_grid = None
        self.input_alpha = -1

        self.kv_data_type = None

        assert self.in_features % self.group_size == 0

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, layer_id, layer_name, init_only=False, ant_config=None):

        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, ant_config, layer_id, layer_name)
        if init_only:  # just prepare for loading sd
            return awq_linear

        awq_linear.weight = linear.weight.data.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        if w_bit > 6:
            awq_linear.ant_config['ant_mode'] = 'int'
        return awq_linear
    
    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input = x.reshape(-1, x.shape[-1])
        # print(input, self.weight)

        # quantize activation to INT8
        input = pseudo_quantize_int(input, n_bit=8, zero_point=True, q_group_size=64)
        
        # print(self.weight_quant_grid, self.weight_alpha, self.input_quant_grid, self.input_alpha)
        # if self.weight_quant_grid is None:

        #     deq_weight = codeant_quant(self, self.w_bit, self.weight, input, self.ant_config, self.group_size, self.layer_id, self.layer_name, is_input=False)

        #     self.weight = deq_weight
        # else:
        #     # deq_weight = get_quant(self.weight, self.weight_quant_grid, alpha=self.weight_alpha, is_input=False)
        #     org_shape = input.shape
        #     deq_input = get_quant(input.view(-1), self.input_quant_grid, alpha=self.input_alpha, is_input=True)
        #     deq_input = deq_input.reshape(org_shape)

        # # quant KV
        # if self.name == 'self_attn.k_proj' or self.name == 'self_attn.v_proj':
        #     pass

        out = F.linear(input, self.weight)

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    