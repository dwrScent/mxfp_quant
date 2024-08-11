
import torch
import torch.nn as nn
from .ant_quant import generate_quant_grid
import torch.nn.functional as F

def int_value(n_bit, signed=True):
    B = n_bit - 1 if signed else n_bit

    values = []
    values.append(0.)
    for i in range(1, 2 ** B):
        values.append(i)
        if signed:
            values.append(-i)
            
    values = torch.tensor(values) 
    values, _ = torch.sort(values)
    # add a bias to normalize the codebook (the threshold between outliers and normal values is 32)
    values *= 32 / (2 ** B)
    return values

def flint_value(n_bit, signed=True, exp_base = 0):
    B = n_bit - 1 if signed else n_bit

    value_bit = B
    assert(value_bit >= 2)

    exp_num =     value_bit * 2 - 1
    neg_exp_num = value_bit - 1
    pos_exp_num = value_bit - 1
    
    exp_max = pos_exp_num + exp_base
    exp_min = -neg_exp_num

    ## zero
    values = [0.]

    ## exponent negtive
    for i in range(0, neg_exp_num + 1):
        exp_bit = i + 2
        exp_value = -(exp_bit - 1)
        mant_bit = value_bit - exp_bit
        for j in range(int(2 ** mant_bit)):
            v = 2 ** exp_value * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if signed:
                values.append(-v)

    ## exponent zero
    exp_bit = 2
    exp_value = 0
    mant_bit = value_bit - exp_bit
    for j in range(int(2 ** mant_bit)):
        v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
        values.append(v)
        if signed:
            values.append(-v)
            
    ## exponent positive     
    for i in range(1, pos_exp_num):
        exp_bit = i + 2
        exp_value = i
        mant_bit = value_bit - exp_bit
        for j in range(int(2 ** mant_bit)):
            v = 2 ** (exp_value + exp_base) * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if signed:
                values.append(-v)
                
    ## max value
    values.append(2 ** exp_max)
    if signed:
        values.append(-2 ** exp_max)
        
    values = torch.tensor(values)
    values, _ = torch.sort(values)
    # add a bias to normalize the codebook (the threshold between outliers and normal values is 32)
    values *= 32 / (2 ** exp_max)

    return values

def outlier_value(n_bit, signed=True, exp_bit=2, exp_base=5):
    B = n_bit - 1 if signed else n_bit
        
    value_bit = B
    mant_bit = value_bit - exp_bit
    values = []
    
    for i in range(exp_base, exp_base + 2 ** exp_bit):
        for j in range(int(2 ** mant_bit)):
            if i == exp_base and j == 0:
                continue

            v = 2 ** i * (1 + 2 ** (-mant_bit) * j)
            values.append(v)
            if signed:
                values.append(-v)

    values = torch.tensor(values)
    values, _ = torch.sort(values)
                
    return values


@torch.no_grad()
def get_quant(tensor_value, quant_grid, outlier_grid, alpha=1.0, group_size=-1):

    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)
    outlier_grid = outlier_grid.to(tensor_value.device)
    merge_grid = torch.cat((quant_grid, outlier_grid), dim=0)

    if group_size == -2:
        tensor_value = tensor_value.view(-1)

        mean = tensor_value.mean()
        std = tensor_value.std()
        normal_max = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs())

        max_quant_val = max(quant_grid)
        scales = (normal_max * alpha) / max_quant_val
        zeros = 0

        batch_num = 32
        assert tensor_value.shape[0] % batch_num == 0
        batch_size = tensor_value.shape[0] // batch_num
        tensor_q = torch.zeros_like(tensor_value)
        for idx in range(batch_num):
            tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size]
            labels = (((tensor_par + zeros) / scales).unsqueeze(-1) - merge_grid).abs().argmin(dim=-1)
            tensor_q_par = merge_grid[labels]
            tensor_q[idx*batch_size : (idx+1)*batch_size] = tensor_q_par

    
    elif group_size >= -1:
        if group_size > 0:
            assert org_shape[-1] % group_size == 0
            tensor_value = tensor_value.reshape(-1, group_size)
        new_shape = tensor_value.shape

        mean = tensor_value.mean(dim=1, keepdim=True)
        std = tensor_value.std(dim=1, keepdim=True)
        normal_max = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs())

        max_quant_val = max(quant_grid)
        scales = (normal_max * alpha) / max_quant_val
        zeros = 0

        # Batch processing to avoid OOM
        batch_num = 32

        assert tensor_value.shape[0] % batch_num == 0
        batch_size = tensor_value.shape[0] // batch_num
        tensor_q = torch.zeros_like(tensor_value)
        for idx in range(batch_num):
            tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
            labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - merge_grid).abs().argmin(dim=-1)
            tensor_q_par = merge_grid[labels]
            tensor_q[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par
        
        tensor_q = tensor_q.view(-1)    

    # Outlier Victim Pair Encoding
    mask = tensor_q.abs() > 32
    victim_odd = torch.roll(mask, 1, -1)
    victim_odd[::2] = 0
    victim_even = torch.roll(mask & (~victim_odd), -1, -1)
    victim_even[1::2] = 0
    victim = victim_even | victim_odd
    tensor_q = tensor_q * (~victim)

    if group_size >= -1:
        tensor_q = tensor_q.view(new_shape)

    tensor_deq = tensor_q * scales - zeros

    tensor_deq = tensor_deq.to(tensor_value.device).half()
    tensor_deq = tensor_deq.reshape(org_shape)

    return tensor_deq

def olive_quant(self, n_bit, weight, input, ant_config, group_size, layer_id, layer_name, exp_base=5, is_input=False):

    # assert group_size == -1, "OliVe use per-channel quantization for weight and per-tensor for activation"
    mode_list = ant_config['ant_mode'].split('-')

    # For MSE computation
    org_output = torch.mm(input, weight.T).to(torch.float)

    int_grid = int_value(n_bit, signed=True)
    flint_grid = flint_value(n_bit, signed=True)
    if n_bit == 8:
        outlier_grid = outlier_value(n_bit, signed=True, exp_bit=4)
    else:
        outlier_grid = outlier_value(n_bit, signed=True, exp_base=exp_base)

    if is_input:
        tensor_value = input
        final_tensor = torch.zeros_like(input, dtype=torch.half).to(input.device)
    else:
        tensor_value = weight
        final_tensor = torch.zeros_like(weight, dtype=torch.half).to(weight.device)

    min_mse = float('inf')
    best_mode = 'null'
    best_alpha = -1
    # Lower bound and upper bound of alpha
    lb = ant_config['w_low']
    ub = ant_config['w_high']

    for idx, mode in enumerate(mode_list):
        # support flint and int in OliVe
        if mode == 'int':
            quant_grid = int_grid
        elif mode == 'flint':
            quant_grid = flint_grid
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        for i in range(lb, ub, 10):
            search_alpha = i * 0.01

            tensor_deq = get_quant(tensor_value, quant_grid, outlier_grid, alpha=search_alpha, group_size=group_size)
            if is_input:
                deq_output = torch.mm(tensor_deq, weight.T).to(torch.float)
            else:
                deq_output = torch.mm(input, tensor_deq.T).to(torch.float)

            mse = (deq_output - org_output).pow(2).mean()

            if mse < min_mse:
                min_mse = mse
                final_tensor = tensor_deq
                best_mode = mode
                best_alpha = search_alpha

    if is_input:
        self.input_quant_grid = int_grid if best_mode == 'int' else flint_grid
        self.input_outlier_grid = outlier_grid
        self.input_alpha = best_alpha
        quant_obj = 'input'
    else:
        self.weight_quant_grid = int_grid if best_mode == 'int' else flint_grid
        self.weight_outlier_grid = outlier_grid
        self.weight_alpha = best_alpha
        quant_obj = 'weight'
    print(f"layer: {layer_id}, tensor: {layer_name}, {quant_obj} quant, best mode: {best_mode}, mse: {min_mse}, alpha: {best_alpha}, bit_width: {n_bit}")
    # if is_input:
    #     print(f"normal_max: {normal_max}, max: {input.max()}, deq_max: {final_tensor.max()}, deq_max  /normal_max: {final_tensor.max() / normal_max}, exp_base: {exp_base}")
    # else:
    #     print(f"normal_max.max(): {normal_max.max()}, max: {weight.max()}, deq_max: {final_tensor.max()}, deq_max  /normal_max.max(): {final_tensor.max() / normal_max.max()}, exp_base: {exp_base}")
    
    if is_input:
        print(f"input max: {tensor_value.max()}, deq_max: {final_tensor.max()}, exp_base: {exp_base}")
    else:
        print(f"weight max: {tensor_value.max()}, deq_max: {final_tensor.max()}, exp_base: {exp_base}")

    return final_tensor

class OliVe_Linear(nn.Module):
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

        # OliVe parameters
        self.weight_quant_grid = None
        self.weight_outlier_grid = None
        self.weight_alpha = -1
        self.input_quant_grid = None
        self.input_outlier_grid = None
        self.input_alpha = -1

        self.tensor_wise_activation = True


        assert self.in_features % self.group_size == 0

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, a_bit, group_size, layer_id, layer_name, init_only=False, ant_config=None):

        awq_linear = cls(w_bit, a_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, ant_config, layer_id, layer_name)
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

        # if 'gate' in self.layer_name or 'q_proj' in self.layer_name or 'up' in self.layer_name or 'k_proj' in self.layer_name:
        # if 'mlp' in self.layer_name:
        #     self.w_bit = 8
        #     self.a_bit = 8

        # Search and set data type and alpha during the first inference
        if self.weight_quant_grid is None:
            if self.group_size > -1:
                # Channel-wise search
                deq_weight = olive_quant(self, self.w_bit, self.weight, input, self.ant_config, -1, self.layer_id, self.layer_name, exp_base=5, is_input=False)

                deq_weight = get_quant(self.weight, self.weight_quant_grid, self.weight_outlier_grid, alpha=self.weight_alpha, group_size=self.group_size)
                self.weight = deq_weight
            else:
                deq_weight = olive_quant(self, self.w_bit, self.weight, input, self.ant_config, -1, self.layer_id, self.layer_name, exp_base=5, is_input=False)
                self.weight = deq_weight

            # Tensor-wise search
            deq_input = olive_quant(self, self.a_bit, deq_weight, input, self.ant_config, -2, self.layer_id, self.layer_name, exp_base=7, is_input=True)
            print("olive search data type and alpha.")
            
        # quantize input based on the selected data type and alpha
        else:
            # OliVe Tensor-wise quantization for activation
            if self.tensor_wise_activation:
                deq_input = get_quant(input, self.input_quant_grid, self.input_outlier_grid, alpha=self.input_alpha, group_size=-2)
            else:
                deq_input = get_quant(input, self.input_quant_grid, self.input_outlier_grid, alpha=self.input_alpha, group_size=self.group_size)

        out = F.linear(deq_input, self.weight)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    