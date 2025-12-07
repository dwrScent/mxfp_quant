# ==============================================================================
# 1. Model Architecture Definitions
#    (Defines shapes: K, N. M is determined at runtime, usually 2048)
# ==============================================================================

from typing import List, NamedTuple


class TensorSpec(NamedTuple):
    """Defines the shape of a single layer (GEMM)."""
    name: str   # e.g., 'q_proj', 'up_proj'
    n: int      # Output dimension (or Weight width)
    k: int      # Input dimension (or Weight height)

class ModelConfig:
    """Defines the architecture of a model (shapes and layer sequence)."""
    def __init__(self, name: str, block: List[TensorSpec]):
        self.name = name
        self.block = block

# Llama-2-7B / Mant-Llama2
# Structure: Q, K, V, O (Attention) + Gate, Up, Down (MLP)
_llama2_7b_block = [
    TensorSpec("q", 4096, 4096),
    TensorSpec("k", 4096, 4096),
    TensorSpec("v", 4096, 4096),
    TensorSpec("o", 4096, 4096),
    TensorSpec("gate", 11008, 4096), # K=4096, N=11008
    TensorSpec("up", 11008, 4096),   # K=4096, N=11008
    TensorSpec("down", 4096, 11008), # K=11008, N=4096
]

# Falcon-7B
# Structure: Attn_Mix, Attn_Out, MLP_Up, MLP_Down
_falcon_7b_block = [
    TensorSpec("attn_mix", 4672, 4544),
    TensorSpec("attn_out", 4544, 4544),
    TensorSpec("mlp_up", 18176, 4544),
    TensorSpec("mlp_down", 4544, 18176),
]

# Llama-3-8B / Mistral-7B (Similar structure with GQA)
# Structure: Q, K, V, O, Gate, Up, Down
_llama3_8b_block = [
    TensorSpec("q", 4096, 4096),
    TensorSpec("k", 1024, 4096),     # GQA: Smaller N
    TensorSpec("v", 1024, 4096),     # GQA: Smaller N
    TensorSpec("o", 4096, 4096),
    TensorSpec("gate", 14336, 4096),
    TensorSpec("up", 14336, 4096),
    TensorSpec("down", 4096, 14336),
]

# Llama-3-70B
_llama3_70b_block = [
    TensorSpec("q", 8192, 8192),
    TensorSpec("k", 1024, 8192),
    TensorSpec("v", 1024, 8192),
    TensorSpec("o", 8192, 8192),
    TensorSpec("gate", 28672, 8192),
    TensorSpec("up", 28672, 8192),
    TensorSpec("down", 8192, 28672),
]

# OPT-6.7B
# Structure: Q, K, V, O, Up, Down
_opt_6b7_block = [
    TensorSpec("q", 4096, 4096),
    TensorSpec("k", 4096, 4096),
    TensorSpec("v", 4096, 4096),
    TensorSpec("o", 4096, 4096),
    TensorSpec("up", 16384, 4096),
    TensorSpec("down", 4096, 16384),
]

models = {
    "llama2_7b":  ModelConfig("llama2_7b",  _llama2_7b_block),
    "falcon_7b":  ModelConfig("falcon_7b",  _falcon_7b_block),
    "llama3_8b":  ModelConfig("llama3_8b",  _llama3_8b_block), 
    "llama3_70b": ModelConfig("llama3_70b", _llama3_70b_block), 
    "mistral_7b": ModelConfig("mistral_7b", _llama3_8b_block), 
    "opt6b7":     ModelConfig("opt6b7",     _opt_6b7_block),
}
