
import torch
import torch.nn as nn
from .ant_quant import generate_quant_grid
from .ant_quant import get_quant_weight
import torch.nn.functional as F
import math

from .quant_func import get_quant_mxfp

from .ant_quant import float_value, int_value, normal_float_value

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

    org_value = tensor_value.clone()

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


    mask_down = torch.where(quant_mse_down < quant_mse_up, torch.tensor(1), torch.tensor(0))
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
    
    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input = x.reshape(-1, x.shape[-1])

        # Search and set data type and alpha in the first inference
        if self.search_tag is None:
            if self.w_bit < 16:
                # deq_weight, _ = get_quant_mxfp(self.weight, quant_grid=self.weight_quant_grid, mode=None, zero_point=False, q_group_size=self.group_size)
                # deq_weight, _ = mxfp_search(self.weight, quant_grid=self.weight_quant_grid, mode=None, zero_point=False, q_group_size=self.group_size)
                # deq_weight, _ = dtype_search(self.weight, quant_grid=self.weight_quant_grid, mode=None, zero_point=False, q_group_size=self.group_size)
                deq_weight, _ = dtype_search_v2(self.weight, quant_grid=self.weight_quant_grid, mode=None, zero_point=False, q_group_size=self.group_size)
            else:
                deq_weight = self.weight
            
            # Quantize weight only once
            self.weight = deq_weight
            deq_input = input
            self.search_tag = 1

        # quantize input based on the selected data type and alpha
        else:
            if self.a_bit < 16:
                # deq_input, _ = get_quant_mxfp(input, quant_grid=self.input_quant_grid, mode=None, zero_point=False, q_group_size=self.group_size)
                # deq_input, _ = mxfp_search(input, quant_grid=self.input_quant_grid, mode=None, zero_point=False, q_group_size=self.group_size)
                # deq_input, _ = dtype_search(input, quant_grid=self.input_quant_grid, mode=None, zero_point=False, q_group_size=self.group_size)
                deq_input, _ = dtype_search_v2(input, quant_grid=self.input_quant_grid, mode=None, zero_point=False, q_group_size=self.group_size)
            else:
                deq_input = input

        out = F.linear(deq_input, self.weight)
        # print('test', self.layer_name, out, out.max(), out.min())

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    