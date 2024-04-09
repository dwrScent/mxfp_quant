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

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, GPT2LMHeadModel):
        layers = model.transformer.h
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    else:
        raise NotImplementedError(type(model))

    return layers
    
