
import torch
import torch.nn as nn
from .ant_quant import generate_quant_grid
import torch.nn.functional as F

from .ant_quant import ant_quantization

def pseudo_quantize_int(tensor, n_bit=8, zero_point=True, q_group_size=-1, alpha=1.0, is_input=False):
    org_shape = tensor.shape
    # assert q_group_size == -1
    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor = tensor.reshape(-1, q_group_size)
    # assert tensor.dim() == 2

    if is_input:
        max_val = tensor.abs().amax()
        max_val = max_val.clamp(min=1e-5)
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

@torch.no_grad()
def get_quant(tensor_value, quant_grid, alpha=1.0, is_input=False):

    org_shape = tensor_value.shape

    # target_device = tensor_value.device
    # if quant_grid.device != target_device:
    #     quant_grid = quant_grid.cpu().to(device=target_device)
    quant_grid = quant_grid.to(tensor_value.device)

    # tensor-wise for activation quantization
    if is_input:
        max_val = tensor_value.abs().amax()
    # channel-wise for weight quantization
    else:
        max_val = tensor_value.abs().amax(dim=1, keepdim=True)

    max_quant_val = max(quant_grid)
    scales = (max_val * alpha) / max_quant_val
    zeros = 0

    # labels = (((tensor_value + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
    # tensor_deq = quant_grid[labels] * scales - zeros
    # argmin, find to index
    if is_input:
        batch_num = 32
        assert org_shape[0] % batch_num == 0
        batch_size = org_shape[0] // batch_num
        tensor_deq = torch.zeros_like(tensor_value)
        for idx in range(batch_num):
            tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size]
            labels = (((tensor_par + zeros) / scales).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
            tensor_q_par = quant_grid[labels] * scales - zeros
            tensor_deq[idx*batch_size : (idx+1)*batch_size] = tensor_q_par
        # tensor_deq = quant_grid[labels] * scales - zeros
    else:
        # Batch processing to avoid OOM
        batch_num = 32
        assert org_shape[0] % batch_num == 0
        batch_size = org_shape[0] // batch_num
        tensor_deq = torch.zeros_like(tensor_value)
        for idx in range(batch_num):
            tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
            labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
            tensor_q_par = quant_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
            tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    # quant_mse = (tensor_deq - tensor_value).abs().pow(2)

    # tensor_deq = tensor_deq.to(tensor_value.device).half()
    tensor_deq = tensor_deq.half()

    return tensor_deq

def ant_quant(self, n_bit, weight, input, ant_config, group_size, layer_id, layer_name, is_input=False):
    # channel-wise quantization for weight, tensor-wise for activation
    # assert group_size == -1, "ANT use per-channel quantization for weight and per-tensor for activation"
    assert torch.isnan(weight).sum() == 0
    assert torch.isnan(input).sum() == 0
    mode_list = ant_config['ant_mode'].split('-')
    quant_grid_set = generate_quant_grid(n_bit=n_bit, signed=True, ant_mode=ant_config['ant_mode'])

    # transform to float to avoid overflow in MSE compute
    # org_output = torch.mm(input, weight.T).to(torch.float64).to(input.device)
    org_output = torch.mm(input, weight.T).to(torch.float64)

    min_mse = float('inf')
    best_mode = 'null'
    best_alpha = -1
    # lower bound and upper bound of alpha
    lb = ant_config['w_low']
    ub = ant_config['w_high']

    # tensor-wise for activation quantization
    if is_input:
        org_shape = input.shape
        final_tensor = torch.zeros_like(input, dtype=torch.half)
        input = input.reshape(-1)
    else:
        final_tensor = torch.zeros_like(weight, dtype=torch.half)

    for idx, mode in enumerate(mode_list):
        quant_grid = quant_grid_set[mode]
        for i in range(lb, ub, 10):
            search_alpha = i * 0.01

            if is_input:
                if n_bit > 6:
                    tensor_deq = pseudo_quantize_int(input, n_bit=n_bit, zero_point=False, q_group_size=group_size, is_input=True)
                else:
                    tensor_deq = get_quant(input, quant_grid, alpha=search_alpha, is_input=True)
                # reshape from (-1) to (seq_len, hidden), for mm operation
                tensor_deq = tensor_deq.reshape(org_shape)
                # transform to float to avoid overflow in MSE compute
                # deq_output = torch.mm(tensor_deq, weight.T).to(torch.float64).to(input.device)
                deq_output = torch.mm(tensor_deq, weight.T).to(torch.float64)
            else:
                if n_bit > 6:
                    tensor_deq = pseudo_quantize_int(weight, n_bit=n_bit, zero_point=False, q_group_size=group_size, is_input=False)
                else:
                    tensor_deq = get_quant(weight, quant_grid, alpha=search_alpha, is_input=False)
                deq_output = torch.mm(input, tensor_deq.T).to(torch.float64)

            mse = (deq_output - org_output).pow(2).mean()

            if mse < min_mse:
                min_mse = mse
                final_tensor = tensor_deq
                best_mode = mode
                best_alpha = search_alpha

    # if layer_id == 7:
    #     print(f'input: {input}, weight: {weight}, deq_output: {deq_output}, org_output: {org_output}')
    #     exit(0)
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

        awq_linear = cls(w_bit, a_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, ant_config, layer_id, layer_name)
        if init_only:  # just prepare for loading state dictionary
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

        # TODO: quantize or not based on a_bit
        outlier_config = {
            "method": "none",  
            "keep_ratio": "-1",  
            "keep_num": 1, 
        }


        # if 'gate' in self.layer_name or 'q_proj' in self.layer_name or 'up' in self.layer_name or 'k_proj' in self.layer_name:
        
        # if 'mlp' in self.layer_name or 'q_proj':
        #     self.w_bit = 8
        #     self.a_bit = 8

        # Search and set data type and alpha during the first inference
        if self.weight_quant_grid is None:
            # print(self.weight.device, input.device)

            if self.group_size > 0:
                if self.w_bit > 6:
                    deq_weight = ant_quant(self, self.w_bit, self.weight, input, self.ant_config, self.group_size, self.layer_id, self.layer_name, is_input=False)
                    # deq_weight = pseudo_quantize_int(self.weight, n_bit=self.w_bit, zero_point=False, q_group_size=self.group_size, alpha=1.0, is_input=False)
                else:
                    deq_weight = ant_quant(self, self.w_bit, self.weight, input, self.ant_config, -1, self.layer_id, self.layer_name, is_input=False)
                    deq_weight = ant_quantization(self.weight, n_bit=self.w_bit, q_group_size=self.group_size, ant_mode=self.weight_mode, outlier_config=outlier_config, display=False)
            else:
                deq_weight = ant_quant(self, self.w_bit, self.weight, input, self.ant_config, -1, self.layer_id, self.layer_name, is_input=False)

            # deq_weight = ant_quant(self, self.w_bit, self.weight, input, self.ant_config, -1, self.layer_id, self.layer_name, is_input=False)
            deq_input = ant_quant(self, self.a_bit, deq_weight, input, self.ant_config, -1, self.layer_id, self.layer_name, is_input=True)

            self.weight = deq_weight            
            print("ant search data type and alpha.")

        # quantize input based on the selected data type and alpha
        else:
            org_shape = input.shape
            
            if self.a_bit > 6:
                if self.group_size > 0:
                    deq_input = pseudo_quantize_int(input, n_bit=self.a_bit, zero_point=False, q_group_size=self.group_size, alpha=1.0, is_input=False)
                else:
                    deq_input = pseudo_quantize_int(input.view(-1), n_bit=self.a_bit, zero_point=False, q_group_size=self.group_size, alpha=self.input_alpha, is_input=True)
            else:
                if self.group_size > 0:
                    #  ant_mode=self.input_mode, same data type for all groups
                    deq_input = ant_quantization(input, n_bit=self.a_bit, q_group_size=self.group_size, ant_mode=self.input_mode, outlier_config=outlier_config, display=False)
                else:
                    deq_input = get_quant(input.view(-1), self.input_quant_grid, alpha=self.input_alpha, is_input=True)
            deq_input = deq_input.reshape(org_shape)

        out = F.linear(deq_input, self.weight)
        # out = F.linear(input, self.weight)

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    