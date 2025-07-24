import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
import copy
from transformers import pytorch_utils

from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from mxq.models.llama_giant import LlamaForCausalLM_giant
from mxq.models.llama_mxfp import LlamaForCausalLM_mxfp
from mxq.models.opt_giant import OPTForCausalLM_giant
from mxq.models.bloom_giant import BloomForCausalLM_giant
from mxq.models.mistal_mxfp import MistralForCausalLM_mxfp

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if isinstance(model, (LlamaForCausalLM, LlamaForCausalLM_giant, LlamaForCausalLM_mxfp)):
        layers = model.model.layers
    elif isinstance(model, (OPTForCausalLM, OPTForCausalLM_giant)):
        layers = model.model.decoder.layers
    elif isinstance(model, GPT2LMHeadModel):
        layers = model.transformer.h
    elif isinstance(model, (BloomForCausalLM, BloomForCausalLM_giant)):
        layers = model.transformer.h
    elif isinstance(model, BertForSequenceClassification):
        layers = model.bert.encoder.layer
    elif isinstance(model, (MistralForCausalLM, MistralForCausalLM_mxfp)):
        layers = model.model.layers
    else:
        raise NotImplementedError(type(model))

    return layers
    
