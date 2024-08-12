
import torch
import torch.nn as nn
from .ant_quant import generate_quant_grid
from .ant_quant import get_quant_weight
import torch.nn.functional as F
import math

from .quant_func import get_quant_mxfp

from .ant_quant import float_value

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


        w_exp_field_map = {3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
        w_exp_field = w_exp_field_map[w_bit]
        mxfp_linear.weight_quant_grid = float_value(w_bit, True, w_exp_field)

        a_exp_field_map = {3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
        a_exp_field = a_exp_field_map[a_bit]
        mxfp_linear.input_quant_grid = float_value(a_bit, True, a_exp_field)

        assert mxfp_linear.group_size == 32
            

        return mxfp_linear
    
    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input = x.reshape(-1, x.shape[-1])

        # Search and set data type and alpha in the first inference
        if self.search_tag is None:
            deq_weight, _ = get_quant_mxfp(self.weight, quant_grid=self.weight_quant_grid, mode=None, zero_point=False, q_group_size=self.group_size)
            
            # Quantize weight only once
            self.weight = deq_weight
            deq_input = input
            self.search_tag = 1

        # quantize input based on the selected data type and alpha
        else:
            deq_input, _ = get_quant_mxfp(input, quant_grid=self.input_quant_grid, mode=None, zero_point=False, q_group_size=self.group_size)

        out = F.linear(deq_input, self.weight)
        # print('test', self.layer_name, out, out.max(), out.min())

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    