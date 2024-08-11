
import torch
import torch.nn as nn
from .ant_quant import generate_quant_grid
from .ant_quant import get_quant_weight
import torch.nn.functional as F
import math

from .quant_func import get_quant_mxfp

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
        
        self.quant_grid = None

        assert self.in_features % self.group_size == 0

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

        if w_bit > 6:
            mxfp_linear.ant_config['ant_mode'] = 'int'
            
        if quant_mode == "fp4_e2m1":
            quant_grid = [0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        elif quant_mode == "fp6_e2m3":  
            quant_grid = [0.0,  0.125,  0.25,  0.375,  0.5,  0.625,  0.75,  0.875,
                                      -0.0, -0.125, -0.25, -0.375, -0.5, -0.625, -0.75, -0.875,
                                      1.0,  1.125,  1.25,  1.375,  1.5,  1.625,  1.75,  1.875,
                                     -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875,
                                      2.0,  2.25,  2.5,  2.75,  3.0,  3.25,  3.5,  3.75,
                                     -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75,
                                      4.0,  4.5,  5.0,  5.5,  6.0,  6.5,  7.0,  7.5,
                                     -4.0, -4.5,  -5.0, -5.5, -6.0, -6.5, -7.0, -7.5]
        elif quant_mode == "fp6_e3m2":
            quant_grid = [0.0, -0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75,
                                      1.0, -1.0, 1.25, -1.25, 1.5, -1.5, 1.75, -1.75,
                                      2.0, -2.0, 2.5, -2.5, 3.0, -3.0, 3.5, -3.5,
                                      4.0, -4.0, 5.0, -5.0, 6.0, -6.0, 7.0, -7.0,
                                      8.0, -8.0, 10.0, -10.0, 12.0, -12.0, 14.0, -14.0,
                                      16.0, -16.0, 20.0, -20.0, 24.0, -24.0, 28.0, -28.0,
                                      32.0, -32.0, 40.0, -40.0, 48.0, -48.0, 56.0, -56.0,
                                      64.0, -64.0, 80.0, -80.0, 96.0, -96.0, 112.0, -112.0]
        mxfp_linear.quant_grid = torch.tensor(quant_grid).to(dtype=torch.half)
        
        return mxfp_linear
    
    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input = x.reshape(-1, x.shape[-1])

        # search and set data type and alpha in the first inference
        if self.weight_quant_grid is None:
            org_w_shape = self.weight.shape
            weight = self.weight.reshape(-1, self.group_size)
            
            deq_weight, _ = get_quant_mxfp(weight, quant_grid=self.quant_grid, mode=None, zero_point=False, q_group_size=-1)
            
            deq_weight = deq_weight.reshape(org_w_shape)
            # quantize weight only once
            self.weight = deq_weight
            deq_input = input
            self.weight_quant_grid = self.quant_grid

        # quantize input based on the selected data type and alpha
        else:
            org_inp_shape = input.shape
            input = input.reshape(-1, self.group_size)
            deq_input, _ = get_quant_mxfp(input, quant_grid=self.quant_grid, mode=None, zero_point=False, q_group_size=-1)
            deq_input = deq_input.reshape(org_inp_shape)

        out = F.linear(deq_input, self.weight)
        # print('test', self.layer_name, out, out.max(), out.min())

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    