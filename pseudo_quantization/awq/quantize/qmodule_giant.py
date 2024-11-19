
import torch
import torch.nn as nn
from .ant_quant import generate_quant_grid
import torch.nn.functional as F

def encode_gen_no_zero(w_bit, return_list=False, a_stride=5):
    """
    Generate the computable codebook from the index list
    Computable codebook: a*index + 2^index * b
    """
    coefficient_list = []
    # 选定系数 a
    for coefficient in range(0, 128, a_stride):
        coefficient_list.append(coefficient)
    # supply some specific data type, merge them after removing duplicates
    supply_list = []
    if a_stride == 10:
        # supply_list = [0, 5, 17, 20]
        # supply_list = [0, 17]
        supply_list = [5, 17]
    merged_list = list(set(coefficient_list + supply_list))

    codebook_dict = {}
    b = 1
    for coefficient in merged_list:
    # for coefficient in coefficient_list:
        codebook_list = []
        # for item in range((2 ** w_bit) // 2 - 1):
        for item in range(2 ** w_bit):
            # 0~15 -> -8~7
            index = item - (2 ** (w_bit-1))
            if index < 0:
                index = (-index) - 1
                codebook_list.append(-(coefficient * index + (2 ** index * b)))
            elif index >= 0:
                assert index < (2 ** w_bit // 2)
                codebook_list.append(coefficient * index + (2 ** index * b))
            else:
                raise ValueError(f"Index {index} out of range")

        codebook_list = torch.tensor(codebook_list).to(dtype=torch.half)
        codebook_list, _ = codebook_list.sort()
        # Normalization
        codebook_list = codebook_list / codebook_list.max()
        # to list if need
        if return_list:
            codebook_list = codebook_list.tolist()

        codebook_dict[f"coefficient_{coefficient}"] = codebook_list
    return codebook_dict

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
        codebook_list, _ = codebook_list.sort()
        
        # Normalization
        codebook_list = codebook_list / codebook_list.max()
        # to list if need
        if return_list:
            codebook_list = codebook_list.tolist()

        codebook_dict[f"coefficient_{coefficient}"] = codebook_list

    return codebook_dict

# core quantization method (simulated quantization)
def pseudo_quantize_int(tensor, n_bit=8, zero_point=False, q_group_size=-1, alpha=1.0):
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
        scales = (max_val * alpha) / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(tensor).sum() == 0

    tensor = (torch.clamp(torch.round(tensor / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(tensor).sum() == 0
    tensor = tensor.reshape(org_shape)
    return tensor
    
class GIANT_Linear(nn.Module):
    def __init__(self, w_bit, a_bit, group_size, in_features, out_features, bias, dev, ant_config, layer_id, layer_name, quant_kv):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.ant_config = ant_config

        self.layer_id = layer_id
        self.layer_name = layer_name

        # CODE-ANT param
        self.weight_coefficient_a = None
        self.weight_quant_grid = None
        self.weight_alpha = -1
        self.input_quant_grid = None
        self.input_alpha = -1

        self.quant_kv = quant_kv

        assert self.in_features % self.group_size == 0

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, a_bit, group_size, layer_id, layer_name, quant_kv, init_only=False, ant_config=None):

        mant_linear = cls(w_bit, a_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, ant_config, layer_id, layer_name, quant_kv)
        if init_only:  # just prepare for loading sd
            return mant_linear

        mant_linear.weight = linear.weight.data.clone().half()
        if linear.bias is not None:
            mant_linear.bias = linear.bias.clone().half()

        if w_bit > 6:
            mant_linear.ant_config['ant_mode'] = 'int'
        
        use_8bit_fusion = False
        # use_8bit_fusion = True
        # If 8-bit fusion is enabled, configure accordingly
        if use_8bit_fusion:
            layer_8bits_name = 'q,k,v,up'  # Example of 8-bit fusion layers
            full_layer_mapping = {
                'q': 'self_attn.q_proj', 'k': 'self_attn.k_proj', 'v': 'self_attn.v_proj', 'o': 'self_attn.out_proj',
                'up': 'mlp.up_proj', 'gate': 'mlp.gate_proj', 'down': 'mlp.down_proj',
                'fc1': 'fc1', 'fc2': 'fc2'
            }
            # Convert the comma-separated names into a list
            specified_8bit_layers = layer_8bits_name.split(',')
            matching_full_names = [full_layer_mapping[layer] for layer in specified_8bit_layers if layer in full_layer_mapping]
            is_8bit_layer = any(full_name == layer_name for full_name in matching_full_names)

            if is_8bit_layer:
                mant_linear.a_bit = 8

        return mant_linear
    
    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input = x.reshape(-1, x.shape[-1])
        # input_init = input.clone().detach()
        # print(input, self.weight)

        # print(self.layer_id, self.layer_name,'before forward move', input.device, self.weight.device)

        # if self.layer_name == 'mlp.up_proj' or self.layer_name == 'mlp.gate_proj' or self.layer_name == 'mlp.down_proj' or self.layer_name == 'self_attn.q_proj' or self.layer_name == 'self_attn.k_proj' or self.layer_name == 'self_attn.v_proj':
        # if self.layer_name == 'mlp.up_proj' or self.layer_name == 'mlp.gate_proj' or self.layer_name == 'mlp.down_proj' or self.layer_name == 'self_attn.q_proj' or self.layer_name == 'self_attn.k_proj':
        # if self.layer_name == 'mlp.up_proj' or self.layer_name == 'mlp.gate_proj' or self.layer_name == 'self_attn.q_proj' or self.layer_name == 'self_attn.k_proj' or self.layer_name == 'self_attn.v_proj':
        # if self.layer_name == 'fc1' or self.layer_name == 'self_attn.q_proj' or self.layer_name == 'self_attn.k_proj' or self.layer_name == 'self_attn.v_proj':
        #     self.a_bit = 8
        
        # if self.layer_id >= 16 and self.layer_name == 'self_attn.out_proj':
        #     self.a_bit = 8


        # quantize activation to INT8
        if self.a_bit < 16 and self.a_bit != -1:
            # input = pseudo_quantize_int(input, n_bit=8, zero_point=False, q_group_size=self.group_size)

            # input = pseudo_quantize_int(input, n_bit=self.a_bit, zero_point=False, q_group_size=self.group_size)
            input = pseudo_quantize_int(input, n_bit=self.a_bit, zero_point=False, q_group_size=-1)

            # best_mse = float('inf')
            # best_alpha = 1.0
            # mse = nn.MSELoss()
            # for alpha_search in range(75, 125, 10):
            #     alpha = alpha_search * 0.01
            #     input_deq = pseudo_quantize_int(input, n_bit=self.a_bit, zero_point=False, q_group_size=self.group_size, alpha=alpha)
            #     quantize_mse = mse(input_deq, input_init)
            #     if quantize_mse < best_mse:
            #         best_mse = quantize_mse
            #         best_alpha = alpha
            # print(f'best alpha:{best_alpha}, best mse:{best_mse}')
            # input = pseudo_quantize_int(input, n_bit=self.a_bit, zero_point=False, q_group_size=self.group_size, alpha=best_alpha)
        
        input = input.to(device=self.weight.device)
        # self.weight = self.weight.to(input.dtype)
        # print(self.layer_id, self.layer_name,'forward move', input.device, self.weight.device)
        out = F.linear(input, self.weight)
        # if self.quant_kv:
        #     if self.layer_name == 'self_attn.k_proj' or self.layer_name == 'self_attn.v_proj':
        #         out = pseudo_quantize_int(out, n_bit=self.w_bit, zero_point=False, q_group_size=self.group_size)

        print(f"layer: {self.layer_id}, tensor: {self.layer_name}, a bit_width: {self.a_bit} group: {self.group_size}")

        out = out + self.bias if self.bias is not None else out

        # print(out.device)
        return out.reshape(out_shape)
    