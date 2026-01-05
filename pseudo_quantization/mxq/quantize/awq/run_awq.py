import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from .prequant import run_awq
from .prequant import apply_awq

model_path = "/cephfs/shared/model/llama-3-8b-hf"
dtype = torch.float16

config = AutoConfig.from_pretrained(model_path)
config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    model_path, use_fast=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)
model.eval().cuda()


w_bit = 4
q_config = {
    "zero_point": True,
    "q_group_size": 128,
}

awq_results = run_awq(
    model,
    tokenizer,
    w_bit=w_bit,
    q_config=q_config,
    n_samples=128,
    seqlen=512,
)


apply_awq(model, awq_results)

