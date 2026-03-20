# coding=utf-8
"""PyTorch PanguProMoE (V2-style) model used by openPangu-R-72B-2512.

This file is injected into local HF cache for repos that reference
`modeling_pangu_moe.py` in config `auto_map` but do not ship it.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_pangu_moe import PanguProMoEConfig


logger = logging.get_logger(__name__)


# Copied/adapted from transformers LLaMA implementations.
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask


class PanguProMoERMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class PanguProMoERotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PanguProMoEMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PanguProMoEAttention(nn.Module):
    def __init__(self, config: PanguProMoEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                "Instantiating PanguProMoEAttention without layer_idx is not recommended when caching is enabled."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_nope_dim = getattr(config, "qk_nope_dim", self.hidden_size // self.num_heads)
        self.qk_rope_dim = getattr(config, "qk_rope_dim", 0)
        self.head_dim = self.qk_nope_dim + self.qk_rope_dim
        self.v_channels = getattr(config, "v_channels", self.head_dim)

        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_key_value_heads * self.head_dim
        self.v_size = self.num_key_value_heads * self.v_channels
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        self.qkv_proj = nn.Linear(self.hidden_size, self.q_size + self.k_size + self.v_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_channels, self.hidden_size, bias=False)
        self.k_layernorm = PanguProMoERMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.param_sink_number = int(getattr(config, "param_sink_number", 0) or 0)
        self.param_sink_with_value = bool(getattr(config, "param_sink_with_value", False))
        if self.param_sink_number > 0:
            self.param_sink_key = nn.Parameter(
                torch.empty(self.param_sink_number, self.num_key_value_heads, self.head_dim)
            )
            if self.param_sink_with_value:
                self.param_sink_value = nn.Parameter(
                    torch.empty(self.param_sink_number, self.num_key_value_heads, self.v_channels)
                )
            else:
                self.register_buffer(
                    "param_sink_value",
                    torch.zeros(self.param_sink_number, self.num_key_value_heads, self.v_channels),
                    persistent=False,
                )

        self.rotary_emb = PanguProMoERotaryEmbedding(
            self.qk_rope_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _split_and_apply_rope(self, query_states, key_states, position_ids, kv_seq_len):
        if self.qk_rope_dim <= 0:
            return query_states, key_states

        q_nope, q_rope = query_states[..., : self.qk_nope_dim], query_states[..., self.qk_nope_dim :]
        k_nope, k_rope = key_states[..., : self.qk_nope_dim], key_states[..., self.qk_nope_dim :]

        cos, sin = self.rotary_emb(q_rope, seq_len=kv_seq_len)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position_ids)
        return torch.cat([q_nope, q_rope], dim=-1), torch.cat([k_nope, k_rope], dim=-1)

    def _append_param_sink(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        if self.param_sink_number <= 0:
            return key_states, value_states, attention_mask

        bsz = key_states.size(0)
        sink_k = self.k_layernorm(self.param_sink_key).permute(1, 0, 2).unsqueeze(0).expand(bsz, -1, -1, -1)
        sink_v = self.param_sink_value.permute(1, 0, 2).unsqueeze(0).expand(bsz, -1, -1, -1)
        key_states = torch.cat([sink_k.to(key_states.dtype), key_states], dim=2)
        value_states = torch.cat([sink_v.to(value_states.dtype), value_states], dim=2)

        if attention_mask is not None:
            sink_mask = torch.zeros(
                attention_mask.shape[0],
                attention_mask.shape[1],
                attention_mask.shape[2],
                self.param_sink_number,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([sink_mask, attention_mask], dim=-1)

        return key_states, value_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.v_channels).transpose(1, 2)

        key_states = self.k_layernorm(key_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("layer_idx must be set when using KV cache")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        query_states, key_states = self._split_and_apply_rope(query_states, key_states, position_ids, kv_seq_len)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states, value_states, attention_mask = self._append_param_sink(key_states, value_states, attention_mask)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_channels)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class PanguProMoESparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = int(getattr(config, "n_routed_experts", getattr(config, "num_experts", 0)))
        self.top_k = int(getattr(config, "num_experts_per_tok", 1))
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))
        self.output_router_logits = bool(getattr(config, "output_router_logits", False))
        self.router_enable_expert_bias = bool(getattr(config, "router_enable_expert_bias", False))

        shared_count = int(getattr(config, "n_shared_experts", 0) or 0)

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        if self.router_enable_expert_bias:
            self.e_score_correction_bias = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))
        else:
            self.register_parameter("e_score_correction_bias", None)

        self.experts = nn.ModuleList(
            [PanguProMoEMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        self.shared_experts = None
        if shared_count > 0:
            self.shared_experts = PanguProMoEMLP(
                config,
                intermediate_size=config.moe_intermediate_size * shared_count,
            )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_dim)

        router_logits = self.gate(flat_states)
        routing_scores = torch.sigmoid(router_logits.float())

        # Expert bias is only for expert selection ranking. Keep aggregation
        # weights from raw positive routing scores to avoid negative/unstable
        # top-k weights that can explode after normalization.
        if self.e_score_correction_bias is not None:
            score_for_choice = routing_scores + self.e_score_correction_bias.unsqueeze(0)
        else:
            score_for_choice = routing_scores

        _, topk_idx = torch.topk(score_for_choice, k=self.top_k, dim=-1)
        topk_scores = torch.gather(routing_scores, dim=-1, index=topk_idx)
        if self.norm_topk_prob:
            denom = topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            topk_scores = topk_scores / denom
        topk_scores = (topk_scores * self.routed_scaling_factor).to(flat_states.dtype)

        final_hidden = torch.zeros_like(flat_states)
        expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = flat_states[token_idx]
            current_hidden = self.experts[expert_idx](current_state) * topk_scores[token_idx, idx, None]
            final_hidden.index_add_(0, token_idx, current_hidden)

        if self.shared_experts is not None:
            final_hidden = final_hidden + self.shared_experts(flat_states)

        return final_hidden.reshape(bsz, seq_len, hidden_dim), router_logits


class PanguProMoEDecoderLayer(nn.Module):
    def __init__(self, config: PanguProMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.first_k_dense_replace = int(getattr(config, "first_k_dense_replace", config.num_hidden_layers))
        self.sandwich_norm = bool(getattr(config, "sandwich_norm", False))

        self.self_attn = PanguProMoEAttention(config, layer_idx)
        if layer_idx < self.first_k_dense_replace:
            self.mlp = PanguProMoEMLP(config, intermediate_size=config.intermediate_size)
            self.is_moe = False
        else:
            self.mlp = PanguProMoESparseMoeBlock(config)
            self.is_moe = True

        self.input_layernorm = PanguProMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PanguProMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.sandwich_norm:
            self.pre_mlp_layernorm = PanguProMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_mlp_layernorm = PanguProMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        if self.sandwich_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
            residual = hidden_states
            hidden_states = self.pre_mlp_layernorm(hidden_states)
        else:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe:
            hidden_states, router_logits = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            router_logits = None

        hidden_states = residual + hidden_states

        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs


class PanguProMoEPreTrainedModel(PreTrainedModel):
    config_class = PanguProMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PanguProMoEDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class PanguProMoEModel(PanguProMoEPreTrainedModel):
    def __init__(self, config: PanguProMoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [PanguProMoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = PanguProMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            # Keep the modern Cache object instead of converting to legacy tuple.
            # Legacy tuple cache breaks this model's attention path on newer transformers.
            next_cache = next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # Compatibility for older/newer transformers cache formats.
        # New format: Cache object with get_seq_length().
        # Legacy format: tuple(layer_past), where layer_past[0] is key tensor.
        past_seen_tokens = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_seen_tokens = past_key_values.get_seq_length()
            elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
                first_layer_past = past_key_values[0]
                if (
                    isinstance(first_layer_past, (tuple, list))
                    and len(first_layer_past) > 0
                    and hasattr(first_layer_past[0], "shape")
                ):
                    past_seen_tokens = first_layer_past[0].shape[-2]
        using_static_cache = isinstance(past_key_values, StaticCache)

        if self._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class PanguProMoEForCausalLM(PanguProMoEPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = PanguProMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.0)
        self.num_experts = int(getattr(config, "n_routed_experts", getattr(config, "num_experts", 0)))
        self.num_experts_per_tok = config.num_experts_per_tok
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
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
