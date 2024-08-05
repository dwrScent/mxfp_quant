import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.opt.configuration_opt import *
from transformers.models.opt.modeling_opt import *
# from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from ..utils.make_distribution import group_dist
from ..utils.cdf_graph import group_cdf, cdf_csv
from ..quantize.ant_quant import get_quant_weight

def pseudo_quantize_int(tensor, n_bit=8, zero_point=False, q_group_size=-1, get_scale=False):
    org_shape = tensor.shape
    padding_size = 0
    
    if q_group_size > 0:
        if org_shape[-1] % q_group_size != 0:
            # Calculate padding size
            padding_size = q_group_size - (org_shape[-1] % q_group_size)
            # Apply padding
            tensor = F.pad(tensor, (0, padding_size), "constant", 0)
            padding_shape = tensor.shape
            assert padding_shape[-1] % q_group_size == 0
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
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(tensor).sum() == 0

    tensor = (torch.clamp(torch.round(tensor / scales) +
                        zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(tensor).sum() == 0

    if padding_size > 0:
        tensor = tensor.reshape(padding_shape)
        tensor = tensor[:, :org_shape[-1]]
    else:
        tensor = tensor.reshape(org_shape)

    if get_scale:
        return tensor, max_val, scales
    else:
        return tensor

def pseudo_quantize_giant(tensor, n_bit=8, zero_point=False, q_group_size=-1, get_scale=False):
    quantized_part_shape = tensor.shape

    if q_group_size > 0:
        quantized_part_group = tensor.reshape(-1, q_group_size)
    else:
        raise ValueError('not support yet')
    max_val = torch.max(torch.abs(quantized_part_group), dim=1, keepdim=True).values
    value_var = torch.var(quantized_part_group / max_val, dim=1, keepdim=True)

    quant_grid_set = {}
    quant_grid_set['coefficient_25'] = torch.tensor([-1.0000, -0.7061, -0.5181, -0.3828, -0.2739, -0.1782, -0.0891, -0.0033, 0.0033,  0.0891,  0.1782,  0.2739,  0.3828,  0.5181,  0.7061,  1.0000])
    quant_grid_set['int'] = torch.tensor([-0., -7., -6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5., 6.,  7.])
    quant_grid_set['coefficient_0'] = torch.tensor([-1.0000, -0.5000, -0.2500, -0.1250, -0.0625, -0.0312, -0.0156, -0.0078, 0.0078,  0.0156,  0.0312,  0.0625,  0.1250,  0.2500,  0.5000,  1.0000])


    quantized_part_group_deq_nf, _ = get_quant_weight(quantized_part_group, quant_grid_set['coefficient_25'], mode='coefficient_25', q_group_size=q_group_size)
    quantized_part_group_deq_int, _ = get_quant_weight(quantized_part_group, quant_grid_set['int'], mode='int', q_group_size=q_group_size)
    quantized_part_group_deq_pot, _ = get_quant_weight(quantized_part_group, quant_grid_set['coefficient_0'], mode='coefficient_0', q_group_size=q_group_size)

    mask_pot = (value_var < 0.05).expand_as(quantized_part_group_deq_pot)
    mask_nf = ((value_var >= 0.05) & (value_var <= 0.25)).expand_as(quantized_part_group_deq_nf)
    mask_int = (value_var > 0.25).expand_as(quantized_part_group_deq_int)

    # 使用 mask 选取对应的量化后 tensor
    quantized_part_group_deq = torch.zeros_like(quantized_part_group)

    quantized_part_group_deq = torch.where(mask_pot, quantized_part_group_deq_pot, quantized_part_group_deq)
    quantized_part_group_deq = torch.where(mask_nf, quantized_part_group_deq_nf, quantized_part_group_deq)
    quantized_part_group_deq = torch.where(mask_int, quantized_part_group_deq_int, quantized_part_group_deq)

    quantized_part_deq = quantized_part_group_deq.reshape(quantized_part_shape)
    quantized_part_deq = quantized_part_deq.to(dtype=tensor.dtype, device=tensor.device)

    assert torch.isnan(quantized_part_deq).sum() == 0

    return quantized_part_deq

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]



class OPTAttention_giant(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OPTConfig,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        def _handle_deprecated_argument(config_arg_name, config, fn_arg_name, kwargs):
            """
            If a the deprecated argument `fn_arg_name` is passed, raise a deprecation
            warning and return that value, otherwise take the equivalent config.config_arg_name
            """
            val = None
            if fn_arg_name in kwargs:
                logging.warning(
                    "Passing in {} to {self.__class__.__name__} is deprecated and won't be supported from v4.38."
                    " Please set it in the config instead"
                )
                val = kwargs.pop(fn_arg_name)
            else:
                val = getattr(config, config_arg_name)
            return val

        self.embed_dim = _handle_deprecated_argument("hidden_size", config, "embed_dim", kwargs)
        self.num_heads = _handle_deprecated_argument("num_attention_heads", config, "num_heads", kwargs)
        self.dropout = _handle_deprecated_argument("attention_dropout", config, "dropout", kwargs)
        self.enable_bias = _handle_deprecated_argument("enable_bias", config, "bias", kwargs)

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)

        # Quantization args
        self.quant_attention = True
        self.group_size = 64
        self.print_stats = True
        # self.print_stats = False
        self.q_bit = 8
        self.k_bit = 4
        self.v_bit = 4
        self.v_group_elem_num = 0
        self.v_update_mode = 'lazy_update'
        self.data_type = 'int'

        # self.v_update_mode = 'immediate_update'
        self.reset_local_vars()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def reset_local_vars(self):
        self.local_max = torch.zeros((self.num_key_value_heads * self.head_dim, 1))
        self.local_scaling_factor = torch.ones((self.num_key_value_heads * self.head_dim, 1))

        self.v_channel_scale = torch.ones((self.num_key_value_heads * self.head_dim, 1))

    def quantize_query_key(self, query_states: torch.Tensor, key_states: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        org_q_shape = query_states.shape
        org_k_shape = key_states.shape

        mse = nn.MSELoss()
        org_q = query_states.clone()
        org_k = key_states.clone()

        # Flatten for quantization
        query_states = query_states.reshape(query_states.shape[1], -1)
        key_states = key_states.reshape(key_states.shape[1], -1)

        # cdf_csv(key_states, -2, self.layer_idx, 'key', 1000, 1, 'cdf_key_tensor.csv')  
        # if self.layer_idx >= 12 and self.layer_idx <= 15:
        #     cdf_csv(key_states, -1, self.layer_idx, 'key', 1000, 64, 'cdf_key_chan.csv')  
        # if self.layer_idx >= 12 and self.layer_idx <= 15:
        #     cdf_csv(key_states, 64, self.layer_idx, 'key', 1000, 512, 'cdf_key_group.csv')

        # Quantize query and key states
        query_states = pseudo_quantize_int(query_states, n_bit=self.q_bit, zero_point=False, q_group_size=group_size)

        if self.data_type == 'giant':
            key_states = pseudo_quantize_giant(key_states, n_bit=self.k_bit, zero_point=False, q_group_size=group_size)
        elif self.data_type == 'int':
            key_states = pseudo_quantize_int(key_states, n_bit=self.k_bit, zero_point=False, q_group_size=group_size)
        else:
            raise ImportError('not support yet')
        # key_states = quantized_part_deq

        # Restore original shapes
        query_states = query_states.reshape(org_q_shape)
        key_states = key_states.reshape(org_k_shape)
        if self.print_stats:
            print(f"mse q: {mse(org_q, query_states)} mse k: {mse(org_k, key_states)} q_bit: {self.q_bit} k_bit: {self.k_bit} group_size: {self.group_size}")
        
        return query_states, key_states
    
    def quantize_value(self, value_states: torch.Tensor, org_v_shape: torch.Tensor, group_size: int, bsz: int, q_len: int) -> torch.Tensor:
        # current `value_states` is the new V cache
        v_seq_len = value_states.shape[2]

        mse = nn.MSELoss()
        org_v = value_states.clone()

        if self.v_update_mode == 'lazy_update':
            # org_v_shape (b, s, n*d)
            if org_v_shape[-2] == 1:
                # decode stage
                # quantize the latest $group_size tokens when (V cache % group_size == 0)
                # value_states.shape (b, n, s, d)
                v_cache_shape = value_states.shape
                
                if v_seq_len % group_size == 0:
                    # recover the shape of value cache and reshape to 2d for quantization -> (s, b*n*d)
                    value_states = value_states.transpose(1, 2).reshape(v_seq_len, -1)
                    value_trans = value_states.t()

                    # quantize the latest V cache group
                    if self.data_type == 'giant':
                        quantized_part_deq = pseudo_quantize_giant(value_trans[:, (v_seq_len-group_size):], n_bit=self.v_bit, zero_point=False, q_group_size=group_size)
                    elif self.data_type == 'int':
                        quantized_part_deq = pseudo_quantize_int(value_trans[:, (v_seq_len-group_size):], n_bit=self.v_bit, zero_point=False, q_group_size=group_size)
                    else:
                        raise ImportError('not support yet')

                    value_trans = torch.cat([value_trans[:, :(v_seq_len-group_size)], quantized_part_deq], dim=1)

                    # reshape
                    value_states = value_trans.t()
                    value_states = value_states.reshape(v_cache_shape[0], v_cache_shape[2], v_cache_shape[1], v_cache_shape[3]).transpose(1, 2)
                else:
                    # quantize the V vector to INT8
                    value_states = value_states.transpose(1, 2).reshape(v_seq_len, -1)
                    value_trans = value_states.t()

                    # quantize the latest V cache group
                    quantized_part = value_trans[:, (v_seq_len-1):]
                    # print(quantized_part.shape, self.v_channel_scale.shape, self.v_channel_scale)

                    max_int = 2 ** (8 - 1) - 1
                    min_int = - 2 ** (8 - 1)

                    quantized_part = (torch.clamp(torch.round(quantized_part /  self.v_channel_scale) +
                                    0, min_int, max_int) - 0) *  self.v_channel_scale
                    assert torch.isnan(quantized_part).sum() == 0

                    # quantized_part = pseudo_quantize_int(value_trans[:, (v_seq_len-1):], n_bit=8, zero_point=False, q_group_size=group_size)
                    # print(quantized_part.shape)
                    value_trans = torch.cat([value_trans[:, :(v_seq_len-1)], quantized_part], dim=1)

                    # reshape
                    value_states = value_trans.t()
                    value_states = value_states.reshape(v_cache_shape[0], v_cache_shape[2], v_cache_shape[1], v_cache_shape[3]).transpose(1, 2)
                
                if self.print_stats:
                    print(f"decode mse v: {mse(org_v, value_states)} v_bit: {self.v_bit} q_group_size: {self.group_size}")

                return value_states
            
            # prefill stage
            elif org_v_shape[-2] > 1:
                # recover the shape of value cache and reshape to 2d for quantization -> (s, b*n*d)
                value_states = value_states.transpose(1, 2).reshape(v_seq_len, -1)
                value_trans = value_states.t().contiguous()

                # cdf_csv(value_trans, -2, self.layer_idx, 'value', 1000, 1, 'cdf_value_tensor.csv')  
                # if self.layer_idx >= 12 and self.layer_idx <= 15:
                #     cdf_csv(value_trans, -1, self.layer_idx, 'value', 1000, 64, 'cdf_value_chan.csv')  
                # if self.layer_idx >= 12 and self.layer_idx <= 15:
                #     cdf_csv(value_trans, 64, self.layer_idx, 'value', 1000, 512, 'cdf_value_group.csv')

                # quantize the V cache, leave the (org_v_shape[-2] % group_size) tokens
                quant_elem_num = (v_seq_len // group_size) * group_size

                quantized_part = value_trans[:, :quant_elem_num]
                # print(self.v_channel_scale.shape, value_trans.shape, value_trans.abs().amax(dim=1, keepdim=True))
                self.v_channel_scale = value_trans.abs().amax(dim=1, keepdim=True) / 127

                if self.data_type == 'giant':
                    quantized_part_deq = pseudo_quantize_giant(quantized_part, n_bit=self.v_bit, zero_point=False, q_group_size=group_size)
                elif self.data_type == 'int':
                    quantized_part_deq = pseudo_quantize_int(quantized_part, n_bit=self.v_bit, zero_point=False, q_group_size=group_size)
                else:
                    raise ImportError('not support yet')
                # # intervals = [0, 0.05, 0.25, 0.5, 1.0]
                # # interval_counts = torch.zeros(len(intervals) - 1, dtype=torch.int32)
                
                # # interval_counts[0] = torch.sum((value_var >= intervals[0]) & (value_var < intervals[1]))
                # # interval_counts[1] = torch.sum((value_var >= intervals[1]) & (value_var < intervals[2]))
                # # interval_counts[2] = torch.sum((value_var >= intervals[2]) & (value_var < intervals[3]))
                # # interval_counts[3] = torch.sum((value_var >= intervals[3]) & (value_var <= intervals[4]))

                # # # 打印区间统计
                # # total = value_var.size(0)
                # # for i in range(len(intervals) - 1):
                # #     count = interval_counts[i].item()
                # #     percentage = (count / total) * 100
                # #     print(f"Interval {intervals[i]} - {intervals[i+1]}: {count} values ({percentage:.2f}%)")
                # # print(f"Normalized variance of value_states:\n{value_var}, {value_var.max()}, {value_var.min()}")
                # # group_dist(quantized_part_group, group_size=group_size, layer_idx=self.layer_idx, layer_name='value', max_fig=1000)

                value_trans = torch.cat([quantized_part_deq, value_trans[:, quant_elem_num:]], dim=1)
                # reshape
                value_states = value_trans.t().contiguous()
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


                # exit(0)

                if self.print_stats:
                    print(f"prefill mse v: {mse(org_v, value_states)}  v_bit: {self.v_bit} q_group_size: {self.group_size}")

                return value_states

            else:
                raise ValueError(f'Error value token shape {org_v_shape}')
        elif self.v_update_mode == 'immediate_update':
            # use our real-time update
            if self.local_max is None or self.local_scaling_factor is None:
                self.reset_local_vars()
                
            if org_v_shape[-2] == 1:
                # decode stage
                # value_states.shape (b, n, s, d)
                v_cache_shape = value_states.shape

                self.local_max = self.local_max.to(value_states.device)
                self.local_scaling_factor = self.local_scaling_factor.to(value_states.device)

                # update the max and scaling factor; then update the tokens in V group if needed
                value_states = value_states.transpose(1, 2).reshape(v_seq_len, -1)
                value_trans = value_states.t()

                # quantize the latest V cache group
                quantized_token = value_trans[:, (v_seq_len-1):]
                # print(f'quantized_token shape: {quantized_token.shape} v group_num: {self.v_group_elem_num}')
                if self.v_group_elem_num == 0:

                    # Initialize the max and scaling factor of group
                    quantized_token, self.local_max, self.local_scaling_factor = pseudo_quantize_int(quantized_token, n_bit=self.v_bit, zero_point=False, q_group_size=group_size, get_scale=True)
                    self.v_group_elem_num += 1
                    value_trans = torch.cat([value_trans[:, :(v_seq_len-self.v_group_elem_num)], quantized_token], dim=1)
                
                elif  self.v_group_elem_num > 0 and self.v_group_elem_num < self.group_size:
                    # Need update the V group
                    self.v_group_elem_num += 1
                    current_max = torch.cat([self.local_max, quantized_token], dim=1)
                    current_max = current_max.abs().amax(dim=1, keepdim=True)
                    update_scale = self.local_max / current_max

                    v_group_token = value_trans[:, (v_seq_len-self.v_group_elem_num):(v_seq_len-1)]

                    v_group_token_q = v_group_token / self.local_scaling_factor
                    v_update_token_q = torch.round(v_group_token_q * update_scale)

                    # new scaling factor
                    self.local_scaling_factor = self.local_scaling_factor / update_scale
                    v_update_token_deq = v_update_token_q * self.local_scaling_factor
                    self.local_max = current_max

                    quantized_token = torch.round(quantized_token / self.local_scaling_factor) * self.local_scaling_factor
                    
                    assert torch.isnan(update_scale).sum() == 0
                    assert torch.isnan(quantized_token).sum() == 0
                    assert torch.isnan(v_update_token_deq).sum() == 0
                    assert torch.isnan(value_trans).sum() == 0

                    
                    value_trans = torch.cat([value_trans[:, :(v_seq_len-self.v_group_elem_num)], v_update_token_deq, quantized_token], dim=1)
                elif self.v_group_elem_num == self.group_size:
                    self.v_group_elem_num = 0

                    # Reset the max and scaling factor of group
                    quantized_token, self.local_max, self.local_scaling_factor = pseudo_quantize_int(quantized_token, n_bit=self.v_bit, zero_point=False, q_group_size=group_size, get_scale=True)
                    self.v_group_elem_num += 1
                    value_trans = torch.cat([value_trans[:, :(v_seq_len-self.v_group_elem_num)], quantized_token], dim=1)
                else:
                    pass
                
                # reshape
                value_states = value_trans.t()
                value_states = value_states.reshape(v_cache_shape[0], v_cache_shape[2], v_cache_shape[1], v_cache_shape[3]).transpose(1, 2)

                assert torch.isnan(value_states).sum() == 0

                if self.print_stats:
                    print(f"decode mse v: {mse(org_v, value_states)} v_bit: {self.v_bit} q_group_size: {self.group_size}")

                return value_states
            
            # prefill stage
            elif org_v_shape[-2] > 1:
                # quantize the V cache, leave the (org_v_shape[-2] % group_size) tokens; setting the max and scaling factor buffer

                # recover the shape of value cache and reshape to 2d for quantization -> (s, b*n*d)
                value_states = value_states.transpose(1, 2).reshape(v_seq_len, -1)
                value_trans = value_states.t()

                # quantize the V cache, leave the (org_v_shape[-2] % group_size) tokens
                quant_elem_num = (v_seq_len // group_size) * group_size
                quantized_part = pseudo_quantize_int(value_trans[:, :quant_elem_num], n_bit=self.v_bit, zero_point=False, q_group_size=group_size)

                # quantize the rest V tokens, set the local_max and local_scaling_factor
                if v_seq_len % group_size != 0:
                    # print(value_trans[:, quant_elem_num:])
                    quantizing_group, self.local_max, self.local_scaling_factor = pseudo_quantize_int(value_trans[:, quant_elem_num:], n_bit=self.v_bit, zero_point=False, q_group_size=group_size, get_scale=True)
                    # print(f'total shape: {value_trans.shape} group: {quantizing_group.shape}, {quantizing_group} local_max: {self.local_max.shape} {self.local_max}, local_scaleL {self.local_scaling_factor.shape}')
                    value_trans = torch.cat([quantized_part, quantizing_group], dim=1)

                    # print(quantizing_group, value_trans[:, quant_elem_num:])
                    self.v_group_elem_num = quantizing_group.shape[-1]
                    # exit(0)
                else:
                    value_trans = quantized_part
                    self.v_group_elem_num = 0

                # reshape
                value_states = value_trans.t()
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                if self.print_stats:
                    print(f"prefill mse v: {mse(org_v, value_states)} v_bit: {self.v_bit} q_group_size: {self.group_size}")
                
                return value_states
            else:
                raise ValueError(f'Error value token shape {org_v_shape}')
        print(f'org v shape: {org_v_shape}')

    def quantize_attention_weights(self, attn_weights, group_size):
        mse = nn.MSELoss()
        org_weight_shape = attn_weights.shape
        org_weights = attn_weights.clone()
        attn_weights = attn_weights.reshape(attn_weights.shape[1], -1)
        attn_weights = pseudo_quantize_int(attn_weights, n_bit=self.q_bit, zero_point=False, q_group_size=group_size)
        attn_weights = attn_weights.reshape(org_weight_shape)
        if self.print_stats:
            print(f"mse atten_weights: {mse(org_weights, attn_weights)}")

        return attn_weights

    def quantize_attention_output(self, attn_output, group_size):
        mse = nn.MSELoss()
        org_outputs_shape = attn_output.shape
        org_outputs = attn_output.clone()
        attn_output = attn_output.reshape(attn_output.shape[1], -1)
        attn_output = pseudo_quantize_int(attn_output, n_bit=self.q_bit, zero_point=False, q_group_size=group_size)
        attn_output = attn_output.reshape(org_outputs_shape)
        if self.print_stats:
            print(f"mse atten_outputs: {mse(org_outputs, attn_output)}")
        return attn_output
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        # query_states, key_states, value_states shape (32, 2048, 128)
        quant_attention = True
        if quant_attention == True:
            print('quant_opt_attention')
            org_q_shape = query_states.shape
            org_k_shape = key_states.shape
            org_v_shape = value_states.shape

            mse = nn.MSELoss()
            org_q = query_states.clone()
            org_k = key_states.clone()
            org_v = value_states.clone()

            k_bit = 4
            v_bit = 4

            query_states = query_states.reshape(query_states.shape[1], -1)
            key_states = key_states.reshape(key_states.shape[1], -1)
            value_states = value_states.reshape(value_states.shape[1], -1)
            query_states = pseudo_quantize_int(query_states, n_bit=8, zero_point=False, q_group_size=64)
            key_states = pseudo_quantize_int(key_states, n_bit=k_bit, zero_point=False, q_group_size=64)
            value_trans = value_states.t()
            value_trans = pseudo_quantize_int(value_trans, n_bit=v_bit, zero_point=False, q_group_size=64)
            value_states = value_trans.t()
            # value_states = pseudo_quantize_int(value_states, n_bit=8, zero_point=False, q_group_size=64)

            query_states = query_states.reshape(org_q_shape)
            key_states = key_states.reshape(org_k_shape)
            value_states = value_states.reshape(org_v_shape)

            print(f"mse q: {mse(org_q, query_states)} mse k: {mse(org_k, key_states)} mse v: {mse(org_v, value_states)} k_bit: {k_bit} v_bit: {v_bit}")

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        if quant_attention == True:
            org_probs_shape = attn_probs.shape
            org_probs = attn_probs.clone()
            attn_probs = attn_probs.reshape(attn_probs.shape[1], -1)
            attn_probs = pseudo_quantize_int(attn_probs, n_bit=8, zero_point=False, q_group_size=64)
            attn_probs = attn_probs.reshape(org_probs_shape)

            print(f"mse atten_weights: {mse(org_probs, attn_probs)}")
        

        attn_output = torch.bmm(attn_probs, value_states)

        # quantize attention output
        if quant_attention == True:
            org_outputs_shape = attn_output.shape
            org_outputs = attn_output.clone()
            attn_output = attn_output.reshape(attn_output.shape[1], -1)
            attn_output = pseudo_quantize_int(attn_output, n_bit=8, zero_point=False, q_group_size=64)
            attn_output = attn_output.reshape(org_outputs_shape)

            print(f"mse atten_outputs: {mse(org_outputs, attn_output)}")

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

OPT_ATTENTION_CLASSES = {
    "eager": OPTAttention_giant,
    "flash_attention_2": OptFlashAttention2,
}

class OPTDecoderLayer_giant(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = OPT_ATTENTION_CLASSES[config._attn_implementation](config=config, is_decoder=True)

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OPTDecoder_giant(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer_giant(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel_giant(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder_giant(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM_giant(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel_giant(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

