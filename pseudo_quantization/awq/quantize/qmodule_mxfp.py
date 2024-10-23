
import torch
import torch.nn as nn
from .ant_quant import generate_quant_grid
from .ant_quant import get_quant_weight
import torch.nn.functional as F
import math

from .quant_func import get_quant_mxfp

from .ant_quant import float_value, int_value, normal_float_value


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
def mxfp_direct(tensor_value, quant_grid, mode="int", zero_point=True, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False, n_bit=4):
    org_shape = tensor_value.shape
    ant_mode = 'float-int-pot'
    # ant_mode = 'float-int'
    mode_list = ant_mode.split('-')
    quant_grid_set = generate_quant_grid(n_bit=n_bit, signed=True, ant_mode=ant_mode)


    w_deq_list = {}
    quant_mse_list = {}

    for mode in mode_list:
        w_deq_list[mode], quant_mse_list[mode] = get_quant_mxfp(tensor_value, quant_grid_set[mode], q_group_size=q_group_size)
        exist_mode = mode
    
    if q_group_size > 0:
        tensor_value = tensor_value.reshape(-1, q_group_size)
        for mode in mode_list:
            w_deq_list[mode] = w_deq_list[mode].reshape(-1, q_group_size)
        
    
    data_type_identify = torch.zeros((tensor_value.shape[0], 1), device=tensor_value.device)
    mapping_list = {}
    for idx, mode in enumerate(mode_list):
        mapping_list[mode] = (idx+1)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    assert torch.isnan(max_val).sum() == 0

    max_val = torch.clamp(max_val, min=1e-5)
    mantissa_judge = max_val / torch.pow(2, torch.floor(torch.log2(max_val)))
    assert torch.isnan(mantissa_judge).sum() == 0

    data_type_identify = torch.where(mantissa_judge < 1.125, mapping_list['pot'], data_type_identify)
    data_type_identify = torch.where((mantissa_judge >= 1.125) & (mantissa_judge < 1.625), mapping_list['float'], data_type_identify)
    data_type_identify = torch.where(mantissa_judge >= 1.625, mapping_list['int'], data_type_identify)

    # print(mantissa_judge, data_type_identify, torch.sum(data_type_identify == 0), mapping_list)


    assert torch.sum(data_type_identify == 0) == 0


    # print(mantissa_judge, data_type_identify)

    data_type_mask = {}
    for mode in mode_list:
        data_type_mask[mode] = (data_type_identify == mapping_list[mode])
    
    assert torch.sum(data_type_mask['int']) + torch.sum(data_type_mask['float']) + torch.sum(data_type_mask['pot']) == tensor_value.shape[0]

    # print(data_type_mask, torch.sum(data_type_mask['int']), torch.sum(data_type_mask['float']), torch.sum(data_type_mask['pot']))

    tensor_deq = torch.zeros_like(tensor_value, dtype=torch.float16, device=tensor_value.device)
    for mode in mode_list:
        tensor_deq = tensor_deq + torch.mul(w_deq_list[mode], data_type_mask[mode])

    mse = nn.MSELoss()
    quant_mse = mse(tensor_value, tensor_deq)
    tensor_deq = tensor_deq.reshape(org_shape)


    return tensor_deq, quant_mse



@torch.no_grad()
def mxfp_search(tensor_value, quant_grid, mode="int", zero_point=True, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False):
    '''
    return : dequantized weight, mse?
    '''
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)

    if pos_value is None or pos_value == True:
        max_quant_val = max(quant_grid)
    elif pos_value == False:
        max_quant_val = abs(min(quant_grid))
    else:
        raise NotImplementedError 
    
    # Compute the scaling factor
    # pow(2, math.floor(math.log2(25)) - math.floor(math.log2(6)))
    exp_down = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    exp_up = torch.ceil(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    # scales = (max_val * alpha) / max_quant_val
    scales_down = torch.pow(2, exp_down)
    scales_up = torch.pow(2, exp_up)

    zeros = 0

    # org_value = tensor_value.clone()

    # Batch processing to avoid OOM
    batch_num = 4
    assert tensor_value.shape[0] % batch_num == 0
    batch_size = tensor_value.shape[0] // batch_num

    tensor_deq_down = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales_down[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        tensor_q_par = quant_grid[labels] * scales_down[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq_down[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    tensor_deq_up = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales_up[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        tensor_q_par = quant_grid[labels] * scales_up[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq_up[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    quant_mse_down = (tensor_deq_down-tensor_value).abs().pow(2).mean(dim=1, keepdim=True).to(torch.float32)
    quant_mse_up = (tensor_deq_up-tensor_value).abs().pow(2).mean(dim=1, keepdim=True).to(torch.float32)


    # mask_down = torch.where(quant_mse_down < quant_mse_up, torch.tensor(1), torch.tensor(0))
    mask_down = torch.where(quant_mse_down < quant_mse_up, torch.tensor(1, device=tensor_value.device, dtype=torch.int), torch.tensor(0, device=tensor_value.device, dtype=torch.int))
    tensor_deq = tensor_deq_down * mask_down + tensor_deq_up * (1 - mask_down)
    quant_mse_sum = quant_mse_down * mask_down + quant_mse_up * (1 - mask_down)
    scales = scales_down * mask_down + scales_up * (1 - mask_down)


    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(quant_mse_down).sum() == 0

    tensor_deq = tensor_deq.reshape(org_shape)

    if get_labels:
        return tensor_deq, quant_mse_sum, labels, quant_grid * scales
    else:
        return tensor_deq, quant_mse_sum

@torch.no_grad()
def dtype_search(tensor_value, quant_grid, mode="int", zero_point=True, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False):
    org_shape = tensor_value.shape
    
    quant_grid_fp4 = float_value(4, True)
    quant_grid_int4 = int_value(4, True)

    tensor_deq_fp4, quant_mse_sum_fp4 = mxfp_search(tensor_value, quant_grid=quant_grid_fp4, q_group_size=q_group_size)
    tensor_deq_int4, quant_mse_sum_int4 = mxfp_search(tensor_value, quant_grid=quant_grid_int4, q_group_size=q_group_size)

    if q_group_size > 0:
        tensor_deq_fp4 = tensor_deq_fp4.reshape(-1, q_group_size)
        tensor_deq_int4 = tensor_deq_int4.reshape(-1, q_group_size)
        
    mask_fp = torch.where(quant_mse_sum_fp4 < quant_mse_sum_int4, torch.tensor(1), torch.tensor(0))

    tensor_deq = tensor_deq_fp4 * mask_fp + tensor_deq_int4 * (1 - mask_fp)
    quant_mse_sum = quant_mse_sum_fp4 * mask_fp + quant_mse_sum_int4 * (1 - mask_fp)

    tensor_deq = tensor_deq.reshape(org_shape)

    return tensor_deq, quant_mse_sum
    
@torch.no_grad()
def dtype_search_v2(tensor_value, quant_grid, mode="int", zero_point=True, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False):
    org_shape = tensor_value.shape
    ant_mode = 'float-int-pot'
    mode_list = ant_mode.split('-')
    quant_grid_set = generate_quant_grid(n_bit=4, signed=True, ant_mode=ant_mode)
    w_deq_list = {}
    quant_mse_list = {}

    for mode in mode_list:
        w_deq_list[mode], quant_mse_list[mode] = mxfp_search(tensor_value, quant_grid_set[mode], q_group_size=q_group_size)
        exist_mode = mode
    
    if q_group_size > 0:
        tensor_value = tensor_value.reshape(-1, q_group_size)
        for mode in mode_list:
            w_deq_list[mode] = w_deq_list[mode].reshape(-1, q_group_size)
        
    
    data_type_identify = torch.zeros_like(quant_mse_list[exist_mode], dtype=torch.int32)
    mapping_list = {}
    for idx, mode in enumerate(mode_list):
        mapping_list[mode] = idx
        if idx == 0:
            compared_mse = quant_mse_list[mode]
        else:
            data_type_identify = torch.where(quant_mse_list[mode] < compared_mse, idx, data_type_identify)
            # update the compared_mse
            compared_mse = torch.where(quant_mse_list[mode] < compared_mse, quant_mse_list[mode], compared_mse)
    data_type_mask = {}
    for mode in mode_list:
        data_type_mask[mode] = (data_type_identify == mapping_list[mode])
    
    # print(data_type_mask['int'].shape, w_deq_list['int'].shape, data_type_identify.shape, quant_mse_list['int'].shape)

    tensor_deq = torch.zeros_like(tensor_value, dtype=torch.float16)
    for mode in mode_list:
        quant_grid_set[mode] = quant_grid_set[mode].to(data_type_mask[mode].device)
        tensor_deq = tensor_deq + torch.mul(w_deq_list[mode], data_type_mask[mode])

    mse = nn.MSELoss()
    quant_mse = mse(tensor_value, tensor_deq)
    tensor_deq = tensor_deq.reshape(org_shape)

    return tensor_deq, quant_mse

@torch.no_grad()
def mxfp_search_olive(tensor_value, quant_grid, mode="int", zero_point=True, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False, exp_base=5):
    '''
    return : dequantized weight, mse?
    '''
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)
    outlier_grid = outlier_value(4, signed=True, exp_base=exp_base)
    outlier_grid = outlier_grid.to(tensor_value.device)
    merge_grid = torch.cat((quant_grid, outlier_grid), dim=0)

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    # max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # mean = tensor_value.mean()
    # std = tensor_value.std()
    # normal_max = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs())
    # max_val = torch.where(max_val < normal_max, max_val, normal_max)


    mean = tensor_value.mean(dim=1, keepdim=True)
    std = tensor_value.std(dim=1, keepdim=True)
    normal_max = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs())
    max_val = normal_max
    # print(normal_max, normal_max.shape)
    # exit(0)

    # print(merge_grid.shape, merge_grid.max(), merge_grid.min(), normal_max, quant_grid, outlier_grid, merge_grid)
    # exit(0)
    max_quant_val = max(quant_grid)
    
    # Compute the scaling factor
    # pow(2, math.floor(math.log2(25)) - math.floor(math.log2(6)))
    exp_down = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    exp_up = torch.ceil(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    # scales = (max_val * alpha) / max_quant_val
    scales_down = torch.pow(2, exp_down)
    scales_up = torch.pow(2, exp_up)

    zeros = 0

    # print(tensor_value)

    # Batch processing to avoid OOM
    batch_num = 4
    assert tensor_value.shape[0] % batch_num == 0
    batch_size = tensor_value.shape[0] // batch_num

    tensor_deq_down = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales_down[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - merge_grid).abs().argmin(dim=-1)
        tensor_q_par = merge_grid[labels] * scales_down[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq_down[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    tensor_deq_up = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales_up[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - merge_grid).abs().argmin(dim=-1)
        tensor_q_par = merge_grid[labels] * scales_up[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq_up[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    quant_mse_down = (tensor_deq_down-tensor_value).abs().pow(2).mean(dim=1, keepdim=True).to(torch.float32)
    quant_mse_up = (tensor_deq_up-tensor_value).abs().pow(2).mean(dim=1, keepdim=True).to(torch.float32)


    # mask_down = torch.where(quant_mse_down < quant_mse_up, torch.tensor(1), torch.tensor(0))
    mask_down = torch.where(quant_mse_down < quant_mse_up, torch.tensor(1, device=tensor_value.device, dtype=torch.int), torch.tensor(0, device=tensor_value.device, dtype=torch.int))
    tensor_deq = tensor_deq_down * mask_down + tensor_deq_up * (1 - mask_down)
    quant_mse_sum = quant_mse_down * mask_down + quant_mse_up * (1 - mask_down)
    scales = scales_down * mask_down + scales_up * (1 - mask_down)


    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(quant_mse_down).sum() == 0

    tensor_q = (tensor_deq + zeros) / scales

    # print(tensor_q, tensor_q.amax(dim=1, keepdim=True), torch.sum(tensor_q > 32).item() , tensor_deq, tensor_deq.shape, scales.shape)
    # exit(0)

    mask = tensor_q.abs() > 32
    victim_odd = torch.roll(mask, 1, -1)
    victim_odd[::2] = 0
    victim_even = torch.roll(mask & (~victim_odd), -1, -1)
    victim_even[1::2] = 0
    victim = victim_even | victim_odd
    tensor_q = tensor_q * (~victim)

    tensor_deq = tensor_q * scales - zeros

    # print(mask, mask.shape, victim_odd, victim_even, victim, victim.shape)

    # print(tensor_q, tensor_q.amax(dim=1, keepdim=True), torch.sum(tensor_q > 32).item() , tensor_deq, tensor_deq.shape, scales.shape)

    tensor_deq = tensor_deq.reshape(org_shape)

    if get_labels:
        return tensor_deq, quant_mse_sum, labels, quant_grid * scales
    else:
        return tensor_deq, quant_mse_sum


@torch.no_grad()
def dtype_search_olive(tensor_value, quant_grid, mode="int", zero_point=True, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False, n_bit=4, exp_base=5):
    org_shape = tensor_value.shape
    ant_mode = 'float-int-pot'
    # ant_mode = 'float-int'
    mode_list = ant_mode.split('-')
    quant_grid_set = generate_quant_grid(n_bit=n_bit, signed=True, ant_mode=ant_mode)

    for mode in mode_list:
        if mode == 'pot':
            quant_grid_set[mode] *= 32 / (2 ** (2 ** (n_bit-1) - 2))
        else:
            quant_grid_set[mode] *= 32 / (2 ** (n_bit-1))

    w_deq_list = {}
    quant_mse_list = {}

    for mode in mode_list:
        # if mode == 'pot':
        #     raise NotImplementedError('need scale pot to sub 32')
        w_deq_list[mode], quant_mse_list[mode] = mxfp_search_olive(tensor_value, quant_grid_set[mode], q_group_size=q_group_size, exp_base=exp_base)
        exist_mode = mode
    
    if q_group_size > 0:
        tensor_value = tensor_value.reshape(-1, q_group_size)
        for mode in mode_list:
            w_deq_list[mode] = w_deq_list[mode].reshape(-1, q_group_size)
        
    
    data_type_identify = torch.zeros_like(quant_mse_list[exist_mode], dtype=torch.int32)
    mapping_list = {}
    for idx, mode in enumerate(mode_list):
        mapping_list[mode] = idx
        if idx == 0:
            compared_mse = quant_mse_list[mode]
        else:
            data_type_identify = torch.where(quant_mse_list[mode] < compared_mse, idx, data_type_identify)
            # update the compared_mse
            compared_mse = torch.where(quant_mse_list[mode] < compared_mse, quant_mse_list[mode], compared_mse)
    data_type_mask = {}
    for mode in mode_list:
        data_type_mask[mode] = (data_type_identify == mapping_list[mode])
    
    # print(data_type_mask['int'].shape, w_deq_list['int'].shape, data_type_identify.shape, quant_mse_list['int'].shape)

    tensor_deq = torch.zeros_like(tensor_value, dtype=torch.float16)
    for mode in mode_list:
        quant_grid_set[mode] = quant_grid_set[mode].to(data_type_mask[mode].device)
        tensor_deq = tensor_deq + torch.mul(w_deq_list[mode], data_type_mask[mode])

    mse = nn.MSELoss()
    quant_mse = mse(tensor_value, tensor_deq)
    tensor_deq = tensor_deq.reshape(org_shape)

    # print(f"tensor shape: {tensor_deq.shape}, mse: {quant_mse}, bit_width: {n_bit}")
    # exit(0)

    return tensor_deq, quant_mse

class MXFP_Linear(nn.Module):
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

        # MXFP param
        self.weight_mxfp_mode = ant_config['weight_mxfp_mode']
        self.input_mxfp_mode = ant_config['input_mxfp_mode']


        assert self.in_features % self.group_size == 0

        self.exp_bit_width = None

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, a_bit, group_size, layer_id, layer_name, init_only=False, ant_config=None, quant_mode=None):

        mxfp_linear = cls(w_bit, a_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device, ant_config, layer_id, layer_name)
        if init_only:  # just prepare for loading sd
            return mxfp_linear

        mxfp_linear.weight = linear.weight.data.clone().half()
        if linear.bias is not None:
            mxfp_linear.bias = linear.bias.clone().half()


        # w_exp_field_map = {3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
        # w_exp_field = w_exp_field_map[w_bit]
        if w_bit < 16:
            mxfp_linear.weight_quant_grid = float_value(w_bit, True)
        # mxfp_linear.weight_quant_grid = int_value(w_bit, True)
        # mxfp_linear.weight_quant_grid = normal_float_value(w_bit, True)

        # a_exp_field_map = {3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
        # a_exp_field = a_exp_field_map[a_bit]
        if a_bit < 16:
            mxfp_linear.input_quant_grid = float_value(a_bit, True)
        # mxfp_linear.input_quant_grid = int_value(a_bit, True)
        # mxfp_linear.input_quant_grid = normal_float_value(a_bit, True)


        assert mxfp_linear.group_size == 32
            
        return mxfp_linear
    
    def _quantize_data(self, data, mode, quant_grid, n_bit, exp_base):
        quantize_methods = {
            'base': lambda: get_quant_mxfp(data, quant_grid=quant_grid, mode=None, zero_point=False, q_group_size=self.group_size),
            'scale_search': lambda: mxfp_search(data, quant_grid=quant_grid, mode=None, zero_point=False, q_group_size=self.group_size),
            'dtype_search': lambda: dtype_search_v2(data, quant_grid=quant_grid, mode=None, zero_point=False, q_group_size=self.group_size),
            'dtype_search_olive': lambda: dtype_search_olive(data, quant_grid=quant_grid, mode=None, zero_point=False, q_group_size=self.group_size, n_bit=n_bit, exp_base=exp_base),
            'naive_adapt': lambda: mxfp_direct(data, quant_grid=quant_grid, mode=None, zero_point=False, q_group_size=self.group_size, n_bit=n_bit),
        }
        return quantize_methods.get(mode, lambda: NotImplementedError(f'not support this mxfp mode: {mode}'))()

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input = x.reshape(-1, x.shape[-1])

        # Search and set data type and alpha in the first inference
        if self.search_tag is None:
            if self.w_bit < 16:
                deq_weight, _ = self._quantize_data(self.weight, self.weight_mxfp_mode, self.weight_quant_grid, self.w_bit, 5)

            else:
                deq_weight = self.weight
            
            # Quantize weight only once
            self.weight = deq_weight

            if self.a_bit < 16:
                deq_input, _ = self._quantize_data(input, self.input_mxfp_mode, self.input_quant_grid, self.a_bit, 7)
            else:
                deq_input = input
                
            self.search_tag = 1

        # quantize input based on the selected data type and alpha
        else:
            if self.a_bit < 16:
                deq_input, _ = self._quantize_data(input, self.input_mxfp_mode, self.input_quant_grid, self.a_bit, 7)

            else:
                deq_input = input

        out = F.linear(deq_input, self.weight)
        # print('test', self.layer_name, out, out.max(), out.min())

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    