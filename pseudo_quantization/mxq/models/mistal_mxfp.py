import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.mistral.modeling_mistral import (
    MISTRAL_INPUTS_DOCSTRING,
    MistralFlashAttention2,
    MistralSdpaAttention,
    MistralPreTrainedModel,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from ..quantize.quant_func import get_quant_mxfp
from ..quantize.quant_func import pseudo_quantize_int, get_quant_grid
from ..quantize.ant_quant import generate_quant_grid, ant_quantization

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"


class MistralRotaryEmbedding_mxfp(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        position_ids = position_ids.to(x.device)
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MistralAttention_mxfp(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = MistralRotaryEmbedding_mxfp(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Quantization args
        self.quant_attention = True
        # self.quant_attention = False
        self.group_size = config.group_size
        self.print_stats = True
        # self.print_stats = False
        self.q_bit = config.q_bit
        self.k_bit = config.k_bit
        self.v_bit = config.v_bit
        # self.v_group_elem_num = 0
        # self.v_update_mode = 'lazy_update'
        self.data_type = "mxfp"
        # self.data_type = 'float'
        # self.data_type = 'int'

        # self.keep_outlier = True
        self.keep_outlier = False

        self.quant_grid_set = generate_quant_grid(
            n_bit=4, signed=True, ant_mode="float"
        )

        self.reset_local_vars()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def reset_local_vars(self):
        self.local_max = torch.zeros((self.num_key_value_heads * self.head_dim, 1))
        self.local_scaling_factor = torch.ones(
            (self.num_key_value_heads * self.head_dim, 1)
        )

        self.v_channel_scale = torch.ones((self.num_key_value_heads * self.head_dim, 1))

    def quantize_query_key(
        self, query_states: torch.Tensor, key_states: torch.Tensor, group_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        org_q_shape = query_states.shape
        org_k_shape = key_states.shape

        mse = nn.MSELoss()
        org_q = query_states.clone()
        org_k = key_states.clone()

        # Flatten for quantization
        query_states = query_states.reshape(query_states.shape[1], -1)
        key_states = key_states.reshape(key_states.shape[1], -1)

        # Quantize query and key states
        if self.q_bit < 16:
            if self.data_type == "float":
                quant_grid_set = generate_quant_grid(self.q_bit, ant_mode="float")
                query_states = get_quant_grid(
                    query_states, quant_grid_set["float"], group_size, 1
                )
            elif self.data_type == "mxfp":
                quant_grid_set = generate_quant_grid(self.q_bit, ant_mode="float")
                query_states, _ = get_quant_mxfp(
                    query_states,
                    quant_grid_set["float"],
                    q_group_size=group_size,
                    keep_outlier=self.keep_outlier,
                )
            else:
                query_states = pseudo_quantize_int(
                    query_states,
                    n_bit=self.q_bit,
                    zero_point=False,
                    q_group_size=group_size,
                )

        if self.k_bit < 16:
            key_pt_quantization = True
            # key_pt_quantization = False
            if key_pt_quantization == False:
                key_states = key_states.t().contiguous()
            if self.data_type == "mxfp":
                quant_grid_set = generate_quant_grid(self.k_bit, ant_mode="float")
                key_states, _ = get_quant_mxfp(
                    key_states,
                    quant_grid_set["float"],
                    q_group_size=group_size,
                    keep_outlier=self.keep_outlier,
                )
            elif self.data_type == "int":
                key_states = pseudo_quantize_int(
                    key_states,
                    n_bit=self.k_bit,
                    zero_point=False,
                    q_group_size=group_size,
                )
            elif self.data_type == "float":
                quant_grid_set = generate_quant_grid(self.k_bit, ant_mode="float")
                key_states = get_quant_grid(
                    key_states, quant_grid_set["float"], group_size, 1
                )
            else:
                raise ImportError("not support yet")
            # key_states = quantized_part_deq
            if key_pt_quantization == False:
                key_states = key_states.t().contiguous()

        # Restore original shapes
        query_states = query_states.reshape(org_q_shape)
        key_states = key_states.reshape(org_k_shape)
        if self.print_stats:
            print(
                f"mse q: {mse(org_q, query_states)} mse k: {mse(org_k, key_states)} q_bit: {self.q_bit} k_bit: {self.k_bit} group_size: {self.group_size} quant_dtype: {self.data_type}"
            )

        return query_states, key_states

    def quantize_value(
        self,
        value_states: torch.Tensor,
        org_v_shape: torch.Tensor,
        group_size: int,
        bsz: int,
        q_len: int,
    ) -> torch.Tensor:
        # current `value_states` is the new V cache
        v_seq_len = value_states.shape[2]

        mse = nn.MSELoss()
        org_v = value_states.clone()
        # org_v_shape (b, s, n*d)
        if org_v_shape[-2] == 1:
            # decode stage
            # quantize the latest $group_size tokens when (V cache % group_size == 0)
            # value_states.shape (b, n, s, d)
            v_cache_shape = value_states.shape

            # recover the shape of value cache and reshape to 2d for quantization -> (s, b*n*d)
            value_states = value_states.transpose(1, 2).reshape(v_seq_len, -1)
            value_trans = value_states.t()

            # quantize the latest V cache group
            if self.data_type == "mxfp":
                quant_grid_set = generate_quant_grid(self.v_bit, ant_mode="float")
                quantized_part_deq, _ = get_quant_mxfp(
                    value_trans[:, (v_seq_len - group_size) :],
                    quant_grid_set["float"],
                    q_group_size=group_size,
                    keep_outlier=self.keep_outlier,
                )
            elif self.data_type == "int":
                quantized_part_deq = pseudo_quantize_int(
                    value_trans[:, (v_seq_len - group_size) :],
                    n_bit=self.v_bit,
                    zero_point=False,
                    q_group_size=group_size,
                )
            elif self.data_type == "float":
                quant_grid_set = generate_quant_grid(self.v_bit, ant_mode="float")
                quantized_part_deq = get_quant_grid(
                    value_trans[:, (v_seq_len - group_size) :],
                    quant_grid_set["float"],
                    group_size,
                    1,
                )
            else:
                raise ImportError("not support yet")

            value_trans = torch.cat(
                [value_trans[:, : (v_seq_len - group_size)], quantized_part_deq], dim=1
            )

            # reshape
            value_states = value_trans.t()
            value_states = value_states.reshape(
                v_cache_shape[0], v_cache_shape[2], v_cache_shape[1], v_cache_shape[3]
            ).transpose(1, 2)

            if self.print_stats:
                print(
                    f"decode mse v: {mse(org_v, value_states)} v_bit: {self.v_bit} q_group_size: {self.group_size} quant_dtype: {self.data_type}"
                )

            return value_states

        # prefill stage
        elif org_v_shape[-2] > 1:
            # recover the shape of value cache and reshape to 2d for quantization -> (s, b*n*d)
            value_states = value_states.transpose(1, 2).reshape(v_seq_len, -1)

            value_pc_quantization = True
            # value_pc_quantization = False
            if value_pc_quantization:
                value_trans = value_states.t().contiguous()
            else:
                value_trans = value_states

            # quantize the V cache, leave the (org_v_shape[-2] % group_size) tokens
            quant_elem_num = (v_seq_len // group_size) * group_size

            if value_pc_quantization:
                quantized_part = value_trans[:, :quant_elem_num]
            # IF no transpose
            else:
                quantized_part = value_trans[:quant_elem_num, :]
            # print(self.v_channel_scale.shape, value_trans.shape, value_trans.abs().amax(dim=1, keepdim=True))
            self.v_channel_scale = value_trans.abs().amax(dim=1, keepdim=True) / 127

            if self.data_type == "mxfp":
                quant_grid_set = generate_quant_grid(self.v_bit, ant_mode="float")
                quantized_part_deq, _ = get_quant_mxfp(
                    quantized_part,
                    quant_grid_set["float"],
                    q_group_size=group_size,
                    keep_outlier=self.keep_outlier,
                )
            elif self.data_type == "int":
                quantized_part_deq = pseudo_quantize_int(
                    quantized_part,
                    n_bit=self.v_bit,
                    zero_point=False,
                    q_group_size=group_size,
                )
            elif self.data_type == "float":
                quant_grid_set = generate_quant_grid(self.v_bit, ant_mode="float")
                quantized_part_deq = get_quant_grid(
                    quantized_part, quant_grid_set["float"], group_size, 1
                )
            else:
                raise ImportError("not support yet")
            if value_pc_quantization:
                value_trans = torch.cat(
                    [quantized_part_deq, value_trans[:, quant_elem_num:]], dim=1
                )
                value_states = value_trans.t().contiguous()
            else:
                value_trans = torch.cat(
                    [quantized_part_deq, value_trans[quant_elem_num:, :]], dim=0
                )
                value_states = value_trans
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            if self.print_stats:
                print(
                    f"Prefill mse v: {mse(org_v, value_states)}  v_bit: {self.v_bit} q_group_size: {self.group_size} quant_dtype: {self.data_type}"
                )

            return value_states

        else:
            raise ValueError(f"Error value token shape {org_v_shape}")

    def quantize_attention_weights(self, attn_weights, group_size):
        mse = nn.MSELoss()
        org_weight_shape = attn_weights.shape
        org_weights = attn_weights.clone()
        attn_weights = attn_weights.reshape(attn_weights.shape[1], -1)
        if self.data_type == "float":
            quant_grid_set = generate_quant_grid(self.q_bit, ant_mode="float")
            attn_weights = get_quant_grid(
                attn_weights, quant_grid_set["float"], group_size, 1
            )
        elif self.data_type == "mxfp":
            quant_grid_set = generate_quant_grid(self.q_bit, ant_mode="float")
            attn_weights, _ = get_quant_mxfp(
                attn_weights,
                quant_grid_set["float"],
                q_group_size=group_size,
                keep_outlier=self.keep_outlier,
            )
        else:
            attn_weights = pseudo_quantize_int(
                attn_weights,
                n_bit=self.q_bit,
                zero_point=False,
                q_group_size=group_size,
            )
        attn_weights = attn_weights.reshape(org_weight_shape)
        if self.print_stats:
            print(
                f"mse atten_weights: {mse(org_weights, attn_weights)} quant_dtype: {self.data_type}"
            )

        return attn_weights

    def quantize_attention_output(self, attn_output, group_size):
        mse = nn.MSELoss()
        org_outputs_shape = attn_output.shape
        org_outputs = attn_output.clone()
        attn_output = attn_output.reshape(attn_output.shape[1], -1)
        if self.data_type == "float":
            quant_grid_set = generate_quant_grid(self.q_bit, ant_mode="float")
            attn_output = get_quant_grid(
                attn_output, quant_grid_set["float"], group_size, 1
            )
        elif self.data_type == "mxfp":
            quant_grid_set = generate_quant_grid(self.q_bit, ant_mode="float")
            attn_output, _ = get_quant_mxfp(
                attn_output,
                quant_grid_set["float"],
                q_group_size=group_size,
                keep_outlier=self.keep_outlier,
            )
        else:
            attn_output = pseudo_quantize_int(
                attn_output, n_bit=self.q_bit, zero_point=False, q_group_size=group_size
            )
        attn_output = attn_output.reshape(org_outputs_shape)
        if self.print_stats:
            print(
                f"mse atten_outputs: {mse(org_outputs, attn_output)} quant_dtype: {self.data_type}"
            )
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.quant_attention == True:
            org_v_shape = value_states.shape
            query_states, key_states = self.quantize_query_key(
                query_states, key_states, self.group_size
            )

            # print(f'qk after quant, {query_states.device}, {key_states.device}')

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if self.quant_attention == True and self.v_bit < 16:
            # print(f'v pre quant, {value_states.device}')
            value_states = self.quantize_value(
                value_states, org_v_shape, self.group_size, bsz, q_len
            )
            # Update the V cache
            if past_key_value is not None:
                past_key_value.value_cache[self.layer_idx] = value_states
            # print(f'v after quant, {value_states.device}')

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attention_mask = attention_mask.to(attn_weights.device)
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # quantize attention weights
        if self.quant_attention == True and self.q_bit < 16:
            attn_weights = self.quantize_attention_weights(
                attn_weights, self.group_size
            )

        attn_output = torch.matmul(attn_weights, value_states)

        # quantize attention output
        if self.quant_attention == True and self.q_bit < 16:
            attn_output = self.quantize_attention_output(attn_output, self.group_size)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttention_mxfp,
    "flash_attention_2": MistralFlashAttention2,
    "sdpa": MistralSdpaAttention,
}


class MistralRMSNorm_mxfp(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # print('norm before operation', self.weight.device, hidden_states.device)

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(
            dtype=input_dtype, device=self.weight.device
        )


ALL_LAYERNORM_LAYERS.append(MistralRMSNorm_mxfp)


class MistralMLP_mxfp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)).to(self.gate_proj.weight.device)
            * self.up_proj(x)
        )


class MistralDecoderLayer_mxfp(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MISTRAL_ATTENTION_CLASSES["eager"](config, layer_idx)

        self.mlp = MistralMLP_mxfp(config)
        self.input_layernorm = MistralRMSNorm_mxfp(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MistralRMSNorm_mxfp(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MistralModel_mxfp(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                MistralDecoderLayer_mxfp(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm_mxfp(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (
            attention_mask is not None
            and self._attn_implementation == "flash_attention_2"
            and use_cache
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLM_mxfp(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel_mxfp(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

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
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
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
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
