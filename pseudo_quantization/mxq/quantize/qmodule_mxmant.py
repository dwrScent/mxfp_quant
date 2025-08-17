
import torch
import torch.nn as nn
from .ant_quant import generate_quant_grid
from .ant_quant import get_quant_weight
import torch.nn.functional as F
import math

from .ant_quant import float_value, int_value, normal_float_value
from .rounding_comp import gemm_with_compensation_gpu

from .utils_stats import calculate_scale_range, calculate_outlier_exp
from ..utils.make_distribution import distri_3d

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
        # Normalization to 2
        codebook_list = codebook_list / codebook_list.max() * 1.75
        # to list if need
        if return_list:
            codebook_list = codebook_list.tolist()

        codebook_dict[f"coefficient_{coefficient}"] = codebook_list
    return codebook_dict

@torch.no_grad()
def get_quant_mxmant(tensor_value, quant_grid, mode="int", zero_point=True, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False, is_input=False, keep_outlier=False, print_stats=False):
    '''
    return : dequantized weight, mse
    '''
    assert torch.isinf(tensor_value).sum() == 0
    assert torch.isnan(tensor_value).sum() == 0

    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    # avoid divide a too small value
    max_val = max_val.clamp(min=1e-5)

    max_quant_val = max(quant_grid)

    assert torch.isinf(max_val).sum() == 0

    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp)

    assert not (scales == 0).any(), "Scale should contain 0 values"
    assert torch.isnan(scales).sum() == 0

    zeros = 0

    # Batch processing to avoid OOM
    # batch_num = 4
    batch_num = 4
    assert tensor_value.shape[0]  % batch_num == 0, \
    f"Batch dimension mismatch! Current tensor shape[0]={tensor_value.shape[0]},  batch_num={batch_num}. " \
    f"The first dimension of tensor ({tensor_value.shape[0]})  must be divisible by batch_num ({batch_num})"
    batch_size = tensor_value.shape[0] // batch_num
    tensor_deq = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        tensor_q_par = quant_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    # tensor_deq = org_value * mask + tensor_deq * (1-mask)
    quant_mse = (tensor_deq-tensor_value).abs().pow(2).to(torch.float32)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)


    assert torch.isinf(tensor_deq).sum() == 0
    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(scales).sum() == 0
    # assert torch.isnan(quant_mse).sum() == 0

    tensor_deq = tensor_deq.reshape(org_shape)

    quant_obj = 'input' if is_input else 'weight'
    if print_stats:
        print(f"Quantization MSE: {quant_mse_sum.mean().item()}, quant_obj: {quant_obj}, keep_outlier: {keep_outlier}")

    if get_labels:
        return tensor_deq, quant_mse_sum, labels, quant_grid * scales
    else:
        return tensor_deq, quant_mse_sum
    

@torch.no_grad()
def mxfp_sub_group_adaptive(tensor_value, quant_grid, sub_group_grid, mode="int", zero_point=True, q_group_size=-1, sub_group_size=1, sub_group_mode=None, alpha=1.0, pos_value=None, get_labels=False, is_input=False, keep_outlier=False, print_stats=False):
    org_shape = tensor_value.shape
    if is_input:
        # ant_mode = 'float'
        ant_mode = 'int'
        # ant_mode = 'float-int-pot-flint'
        mode_list = ant_mode.split('-')
        quant_grid_set = generate_quant_grid(n_bit=4, signed=True, ant_mode=ant_mode)
    else:
        quant_grid_set = encode_gen_no_zero(4, a_stride=5)
        int_grid_set = generate_quant_grid(n_bit=4, signed=True, ant_mode='int')
        mode_list = []
        # mode_list = ant_config['ant_mode'].split('-')
        mode_list.extend(quant_grid_set.keys())
        mode_list.append('int')
        quant_grid_set['int'] = int_grid_set['int']  

    w_deq_list = {}
    quant_mse_list = {}

    # sub_group_size = 4

    for mode in mode_list:
        w_deq_list[mode], _ = get_quant_mxmant(tensor_value, quant_grid_set[mode], q_group_size=q_group_size, keep_outlier=keep_outlier, print_stats=print_stats)
        exist_mode = mode

        quant_mse = (w_deq_list[mode]-tensor_value).abs().pow(2).to(torch.float32)
        quant_mse = quant_mse.reshape(-1, sub_group_size)
        quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)
        quant_mse_list[mode] = quant_mse_sum
    
    if sub_group_size > 0:
        tensor_value = tensor_value.reshape(-1, sub_group_size)
        for mode in mode_list:
            w_deq_list[mode] = w_deq_list[mode].reshape(-1, sub_group_size)
    
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
    
    tensor_deq = torch.zeros_like(tensor_value, dtype=torch.float16)
    for mode in mode_list:
        quant_grid_set[mode] = quant_grid_set[mode].to(data_type_mask[mode].device)
        tensor_deq = tensor_deq + torch.mul(w_deq_list[mode], data_type_mask[mode])

    mse = nn.MSELoss()
    quant_mse = mse(tensor_value, tensor_deq)
    tensor_deq = tensor_deq.reshape(org_shape)

    return tensor_deq, quant_mse


class MXMANT_Linear(nn.Module):
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

        # MXFP param
        self.weight_mxfp_mode = ant_config['weight_mxfp_mode']
        self.input_mxfp_mode = ant_config['input_mxfp_mode']

        self.weight_sub_group_size = ant_config.get('weight_sub_group_size')
        self.weight_sub_group_mode = ant_config.get('weight_sub_group_mode')
        self.input_sub_group_size = ant_config.get('input_sub_group_size')
        self.input_sub_group_mode = ant_config.get('input_sub_group_mode')
        self.topk = ant_config.get("topk")
        self.em_bit = ant_config.get("em_bit")
        self.es_bit = ant_config.get("es_bit")
        self.ee_bit = ant_config.get("ee_bit")

        assert self.in_features % self.group_size == 0

        self.exp_bit_width = None

        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))

        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, a_bit, group_size, layer_id, layer_name, init_only=False, ant_config=None, quant_mode=None):

        in_features = linear.weight.shape[1] 
        out_features = linear.weight.shape[0]

        mxfp_linear = cls(w_bit, a_bit, group_size, in_features, out_features, linear.bias is not None, linear.weight.device, ant_config, layer_id, layer_name)
        if init_only:  # just prepare for loading sd
            return mxfp_linear

        # mxfp_linear.weight = linear.weight.data.clone().half()
        mxfp_linear.weight = linear.weight.data
        
        if linear.bias is not None:
            mxfp_linear.bias = linear.bias.clone().half()


        # w_exp_field_map = {3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
        # w_exp_field = w_exp_field_map[w_bit]
        flint_r_list = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5]

        if w_bit < 16:
            if ant_config['ant_mode'] == 'int':
                mxfp_linear.weight_quant_grid = int_value(w_bit, True)
            elif ant_config['ant_mode'] == 'float':
                mxfp_linear.weight_quant_grid = float_value(w_bit, True)
            else:
                raise NotImplementedError('Not support yet.')
            # mxfp_linear.weight_quant_grid = torch.tensor(flint_r_list)
        # mxfp_linear.weight_quant_grid = normal_float_value(w_bit, True)

        # a_exp_field_map = {3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
        # a_exp_field = a_exp_field_map[a_bit]
        if a_bit < 16:
            if ant_config['ant_mode'] == 'int':
                mxfp_linear.input_quant_grid = int_value(a_bit, True)
            elif ant_config['ant_mode'] == 'float':
                mxfp_linear.input_quant_grid = float_value(a_bit, True)
            else:
                raise NotImplementedError('Not support yet.')
            # mxfp_linear.input_quant_grid = torch.tensor(flint_r_list)
        # mxfp_linear.input_quant_grid = normal_float_value(a_bit, True)

        assert mxfp_linear.group_size == 32
            
        return mxfp_linear
    
    def _quantize_data(self, data, mode, quant_grid, n_bit, exp_base, is_input, sub_group_size, sub_group_mode):
        # sub group with E0M3
        sub_group_grid = [0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        # sub_group_grid = [0, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -7.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]
        sub_group_grid = torch.tensor(sub_group_grid)    
        quantize_methods = {
            'base': lambda: get_quant_mxmant(data, quant_grid=quant_grid, mode=None, zero_point=False, q_group_size=self.group_size, is_input=is_input, keep_outlier=self.keep_outlier, print_stats=self.print_stats),
            'sub_group_adaptive': lambda: mxfp_sub_group_adaptive(data, quant_grid=quant_grid, sub_group_grid=sub_group_grid, mode=None, zero_point=False, q_group_size=self.group_size, sub_group_size=sub_group_size, sub_group_mode=sub_group_mode, is_input=is_input, print_stats=self.print_stats),
        }
        return quantize_methods.get(mode, lambda: NotImplementedError(f'not support this mxfp mode: {mode}'))()

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        input = x.reshape(-1, x.shape[-1])

        # Search and set data type and alpha in the first inference
        if self.search_tag is None:
            if self.w_bit < 16:
                deq_weight, _ = self._quantize_data(self.weight, self.weight_mxfp_mode, self.weight_quant_grid, self.w_bit, 5, False, self.weight_sub_group_size, self.weight_sub_group_mode)
            else:
                deq_weight = self.weight
            
            # Quantize weight only once
            self.weight = deq_weight

            if self.a_bit < 16:
                deq_input, _ = self._quantize_data(input, self.input_mxfp_mode, self.input_quant_grid, self.a_bit, 7, True, self.input_sub_group_size, self.input_sub_group_mode)
            else:
                deq_input = input 
            self.search_tag = 1

        # quantize input based on the selected data type and alpha
        else:
            if self.a_bit < 16:
                deq_input, _ = self._quantize_data(input, self.input_mxfp_mode, self.input_quant_grid, self.a_bit, 7, True, self.input_sub_group_size, self.input_sub_group_mode)
                pass
            else:
                deq_input = input

        out = F.linear(deq_input.to(self.weight.dtype), self.weight)
        assert torch.isnan(out).sum() == 0
        if self.print_stats:
            print(f"layer: {self.layer_id}, tensor: {self.layer_name}, a_bit_width: {self.a_bit}. group_size: {self.group_size}")

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    