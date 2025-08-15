
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_func import get_quant_smxfp


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
                                  1.0000, -1.0000,  1.2500, -1.2500,  1.5000, -1.5000,  1.7500, -1.7500]
                                ,dtype=linear.weight.data.dtype)

        fp6_e1m4 = torch.tensor([ 0.0000, -0.0000,  0.0625, -0.0625,  0.1250, -0.1250,  0.1875, -0.1875, 
                                0.2500, -0.2500,  0.3125, -0.3125,  0.3750, -0.3750,  0.4375, -0.4375,
                                0.5000, -0.5000,  0.5625, -0.5625,  0.6250, -0.6250,  0.6875, -0.6875,
                                0.7500, -0.7500,  0.8125, -0.8125,  0.8750, -0.8750,  0.9375, -0.9375,
                                1.0000, -1.0000,  1.0625, -1.0625,  1.1250, -1.1250,  1.1875, -1.1875,
                                1.2500, -1.2500,  1.3125, -1.3125,  1.3750, -1.3750,  1.4375, -1.4375,
                                1.5000, -1.5000,  1.5625, -1.5625,  1.6250, -1.6250,  1.6875, -1.6875,
                                1.7500, -1.7500,  1.8125, -1.8125,  1.8750, -1.8750,  1.9375, -1.9375]
                                ,dtype=linear.weight.data.dtype)
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
    