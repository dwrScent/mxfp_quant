import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.bloom.configuration_bloom import *
from transformers.models.bloom.modeling_bloom import *
# from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

_CHECKPOINT_FOR_DOC = "bigscience/bloom-560m"
_CONFIG_FOR_DOC = "BloomConfig"

BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigscience/bigscience-small-testing",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloom",
]

def pseudo_quantize_int(tensor, n_bit=8, zero_point=False, q_group_size=-1):
    org_shape = tensor.shape
    if q_group_size > 0:
        # assert org_shape[-1] % q_group_size == 0, f'q_group_size {q_group_size} should be a divisor of {org_shape[-1]}'
        if org_shape[-1] % q_group_size != 0:
            # return tensor
            padding = (q_group_size - org_shape[-1] % q_group_size) % q_group_size
            # Padding is only added to the last dimension
            tensor = torch.nn.functional.pad(tensor, (0, padding), "constant", 0)
            # torch.set_printoptions(profile="full")
            print(f'padding last dimension from {org_shape[-1]} to {tensor.shape[-1]}')
            # print(tensor[-1], org_shape, tensor.shape)
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


    # Remove padding to restore original shape
    if org_shape[-1] % q_group_size != 0:
        tensor = tensor.reshape(-1, org_shape[-1] + padding)
        # print(tensor.shape)
        tensor = tensor[:, :org_shape[-1]]  
        # print(tensor.shape)
        # exit(0)

    else:
        tensor = tensor.reshape(org_shape)

    assert tensor.shape == org_shape
    
    return tensor


class BloomAttention_giant(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        # q, v: (b * num_head, seq_len, head_dim), (32, 2048, 128)
        quant_attention = True
        # 这个时候 qkv 应该都是按照行优先排布的
        if quant_attention == True:
            print('quant_bloom_attention')
            org_q_shape = query_layer.shape
            org_k_shape = key_layer.shape
            org_v_shape = value_layer.shape

            mse = nn.MSELoss()
            org_q = query_layer.clone()
            org_k = key_layer.clone()
            org_v = value_layer.clone()

            k_bit = 4
            v_bit = 4

            assert value_layer.shape[-1] > 64
            # query_layer = query_layer.reshape(-1, query_layer.shape[-1])
            # key_layer = key_layer.reshape(-1, key_layer.shape[-1])
            # value_layer = value_layer.reshape(-1, value_layer.shape[-1])
            query_layer = query_layer.reshape(query_layer.shape[1], -1)
            key_layer = key_layer.reshape(key_layer.shape[1], -1)
            value_layer = value_layer.reshape(value_layer.shape[1], -1)
            query_layer = pseudo_quantize_int(query_layer, n_bit=8, zero_point=False, q_group_size=64)
            key_layer = pseudo_quantize_int(key_layer, n_bit=k_bit, zero_point=False, q_group_size=64)

            # 从 input channel 维度量化 V 
            value_trans = value_layer.t()
            # value_trans = value_trans.reshape(org_v_shape[1], -1)
            # print(org_v_shape, value_trans.shape)
            value_trans = pseudo_quantize_int(value_trans, n_bit=v_bit, zero_point=False, q_group_size=64)
            value_layer = value_trans.t()

            # value_layer = pseudo_quantize_int(value_layer, n_bit=4, zero_point=False, q_group_size=64)


            query_layer = query_layer.reshape(org_q_shape)
            key_layer = key_layer.reshape(org_k_shape)
            value_layer = value_layer.reshape(org_v_shape)

            print(f"mse q: {mse(org_q, query_layer)}, mse k: {mse(org_k, key_layer)}, mse v: {mse(org_v, value_layer)} k_bit: {k_bit} v_bit: {v_bit}")


        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attention_probs_reshaped shape (32, 2048, 2048)
        if quant_attention == True:
            org_probs_shape = attention_probs.shape
            org_probs = attention_probs.clone()
            # assert attention_probs.shape[-1] > 64
            # attention_probs = attention_probs.reshape(-1, attention_probs.shape[-1])
            attention_probs = attention_probs.reshape(attention_probs.shape[1], -1)
            attention_probs = pseudo_quantize_int(attention_probs, n_bit=8, zero_point=False, q_group_size=64)
            attention_probs = attention_probs.reshape(org_probs_shape)

            print(f"mse probs: {mse(org_probs, attention_probs)}")

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)


 

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # org_context_shape shape (32, 2048, 2048)
        if quant_attention == True:
            org_context_shape = context_layer.shape
            org_cont = context_layer.clone()

            # assert context_layer.shape[-1] > 64
            # context_layer = context_layer.reshape(-1, context_layer.shape[-1])
            context_layer = context_layer.reshape(context_layer.shape[1], -1)
            context_layer = pseudo_quantize_int(context_layer, n_bit=8, zero_point=False, q_group_size=64)
            context_layer = context_layer.reshape(org_context_shape)

            print(f"mse cont: {mse(org_cont, context_layer)}")


        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs

class BloomBlock_giant(nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = BloomAttention_giant(config)
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions

@add_start_docstrings(
    "The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.",
    BLOOM_START_DOCSTRING,
)
class BloomModel_giant(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Transformer blocks
        self.h = nn.ModuleList([BloomBlock_giant(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        causal_mask = causal_mask.bool()

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

@add_start_docstrings(
    """
    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    BLOOM_START_DOCSTRING,
)
class BloomForCausalLM_giant(BloomPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.transformer = BloomModel_giant(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last tokens for input_ids if past is not None
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

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

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_bloom_cache(reordered_past)