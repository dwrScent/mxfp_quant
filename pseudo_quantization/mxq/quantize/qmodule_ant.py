
import torch
import torch.nn as nn
from .ant_quant import generate_quant_grid
import torch.nn.functional as F

from .ant_quant import ant_quantization

from .quant_func import get_quant_grid

def pseudo_quantize_int(tensor, n_bit=8, zero_point=True, q_group_size=-1, alpha=1.0):
    org_shape = tensor.shape

    # tensor-wise quantization
    if q_group_size == -2:
        tensor = tensor.view(-1)
        max_val = tensor.abs().amax()
        max_val = max_val.clamp(min=1e-5)
    # channel- and group-wise
    elif q_group_size >= -1:    
        if q_group_size > 0:
            assert org_shape[-1] % q_group_size == 0
            tensor = tensor.reshape(-1, q_group_size)
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

def ant_quant(self, n_bit, weight, input, ant_config, group_size, layer_id, layer_name, is_input=False):
    # channel-wise quantization for weight, tensor-wise for activation
    # assert group_size == -1, "ANT use per-channel quantization for weight and per-tensor for activation"
    assert torch.isnan(weight).sum() == 0
    assert torch.isnan(input).sum() == 0
    mode_list = ant_config['ant_mode'].split('-')
    quant_grid_set = generate_quant_grid(n_bit=n_bit, signed=True, ant_mode=ant_config['ant_mode'])

    # Transform to float to avoid overflow in MSE compute
    org_output = torch.mm(input, weight.T).to(torch.float64)

    min_mse = float('inf')
    best_mode = 'null'
    best_alpha = -1
    # Lower bound and upper bound of alpha
    lb = ant_config['w_low']
    ub = ant_config['w_high']

    # Tensor-wise for activation quantization
    if is_input:
        tensor_value = input
        final_tensor = torch.zeros_like(input, dtype=torch.half)
    else:
        tensor_value = weight
        final_tensor = torch.zeros_like(weight, dtype=torch.half)

    for idx, mode in enumerate(mode_list):
        quant_grid = quant_grid_set[mode]
        for i in range(lb, ub, 10):
            search_alpha = i * 0.01

            if n_bit > 6:
                assert mode == 'int'
                tensor_deq = pseudo_quantize_int(tensor_value, n_bit=n_bit, zero_point=False, q_group_size=group_size)
            else:
                tensor_deq = get_quant_grid(tensor_value, quant_grid, group_size=group_size, alpha=search_alpha)

            if is_input:
                deq_output = torch.mm(tensor_deq, weight.T).to(torch.float64)
            else:
                deq_output = torch.mm(input, tensor_deq.T).to(torch.float64)

            mse_cal = nn.MSELoss()
            mse = mse_cal(deq_output, org_output)

            if mse < min_mse:
                min_mse = mse
                final_tensor = tensor_deq
                best_mode = mode
                best_alpha = search_alpha

    if is_input:
        self.input_quant_grid = quant_grid_set[best_mode]
        self.input_alpha = best_alpha
        self.input_mode = best_mode
        quant_obj = 'input'
    else:
        self.weight_quant_grid = quant_grid_set[best_mode]
        self.weight_alpha = best_alpha
        self.weight_mode = best_mode
        quant_obj = 'weight'

    print(f"layer: {layer_id}, tensor: {layer_name}, {quant_obj} quant, best mode: {best_mode}, mse: {min_mse}, alpha: {best_alpha}, bit_width: {n_bit}")
    return final_tensor
class ANT_Linear(nn.Module):
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
        self.weight_mode = None
        self.weight_alpha = -1
        self.input_quant_grid = None
        self.input_mode = None
        self.input_alpha = -1
        assert self.in_features % self.group_size == 0

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, a_bit, group_size, layer_id, layer_name, init_only=False, ant_config=None):

        ant_linear = cls(w_bit, a_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, ant_config, layer_id, layer_name)
        if init_only:  # just prepare for loading state dictionary
            return ant_linear

        # ant_linear.weight = linear.weight.data.clone().half()
        ant_linear.weight = linear.weight.data
        if linear.bias is not None:
            ant_linear.bias = linear.bias.clone().half()

        if w_bit > 6:
            ant_linear.ant_config['ant_mode'] = 'int'
    
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
                ant_linear.a_bit = 8
                ant_linear.w_bit = 8
                ant_linear.ant_config['ant_mode'] = 'int'

        return ant_linear
    
    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input = x.reshape(-1, x.shape[-1])

        # TODO: quantize or not based on a_bit
        outlier_config = {
            "method": "none",  
            "keep_ratio": "-1",  
            "keep_num": 1, 
        }

        self.ant_config['w_low'] = 100
        self.ant_config['w_high'] = 105

        # Search and set data type and alpha during the first inference
        if self.weight_quant_grid is None:
            if self.w_bit > 6:
                # ant_mode is INT, only for search the alpha
                deq_weight = ant_quant(self, self.w_bit, self.weight, input, self.ant_config, self.group_size, self.layer_id, self.layer_name, is_input=False)
                if self.group_size > -1:
                    # ant_mode is INT, alpha is 1.0
                    quant_grid_set = generate_quant_grid(self.w_bit, ant_mode=self.weight_mode)
                    deq_weight = get_quant_grid(self.weight, quant_grid_set[self.weight_mode], self.group_size, alpha=1.0)
            else:
                # search the mode for weight
                deq_weight = ant_quant(self, self.w_bit, self.weight, input, self.ant_config, self.group_size, self.layer_id, self.layer_name, is_input=False)
                if self.group_size > -1:
                    # apply alpha=1.0 in group quantization
                    quant_grid_set = generate_quant_grid(self.w_bit, ant_mode=self.weight_mode)
                    deq_weight = get_quant_grid(self.weight, quant_grid_set[self.weight_mode], self.group_size, alpha=1.0)

            # Tensor-wise search
            deq_input = ant_quant(self, self.a_bit, deq_weight, input, self.ant_config, -2, self.layer_id, self.layer_name, is_input=True)
            # deq_input = ant_quant(self, self.a_bit, deq_weight, input, self.ant_config, self.group_size, self.layer_id, self.layer_name, is_input=True)

            # NOTE: pass the quantized and dequantized input in our experiment in search
            deq_input = input
            self.weight = deq_weight            
            print("ant search data type and alpha.")

        # Quantize input based on the selected data type and alpha
        else:
            org_shape = input.shape
            # ANT falls back to INT when the bit width >= 6
            if self.a_bit < 16:
                if self.a_bit > 6:
                    # Channel-wise for weight and Tensor-wise for activation (ANT init configuration)
                    if self.group_size == -1:
                        deq_input = pseudo_quantize_int(input, n_bit=self.a_bit, zero_point=False, q_group_size=-2, alpha=self.input_alpha)
                    elif self.group_size > 0:
                        # Do not use alpha in group-wise quantization, set it to 1.0
                        deq_input = pseudo_quantize_int(input, n_bit=self.a_bit, zero_point=False, q_group_size=self.group_size, alpha=1.0)
                    else:
                        raise NotImplementedError('Not supported yet')
                else:
                    # Channel-wise for weight and Tensor-wise for activation (ANT init configuration)
                    if self.group_size == -1:
                        deq_input = get_quant_grid(input, self.input_quant_grid, -2, alpha=self.input_alpha)
                    elif self.group_size > 0:
                        quant_grid_set = generate_quant_grid(self.a_bit, ant_mode=self.input_mode)
                        deq_input = get_quant_grid(input, quant_grid_set[self.input_mode], self.group_size, alpha=1.0)
                    else:
                        raise NotImplementedError('Not supported yet')
            else:
                deq_input = input

            deq_input = deq_input.reshape(org_shape)

        out = F.linear(deq_input, self.weight)

        print(f"layer: {self.layer_id}, tensor: {self.layer_name}, a_bit_width: {self.a_bit}. group_size: {self.group_size}")

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    