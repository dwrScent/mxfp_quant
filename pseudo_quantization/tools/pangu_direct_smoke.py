#!/usr/bin/env python
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mxq.evaluation.inference import _ensure_pangu_modeling, _is_openpangu_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct smoke test for openPangu-R-72B-2512 generation")
    parser.add_argument("--model", default="FreedomIntelligence/openPangu-R-72B-2512")
    parser.add_argument("--prompt", default="What is 1+1? Please answer briefly.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    resolved_model = _ensure_pangu_modeling(args.model)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt")
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][input_len:]

    decoded_skip_true = tok.decode(gen_ids, skip_special_tokens=True)
    decoded_skip_false = tok.decode(gen_ids, skip_special_tokens=False)

    report = {
        "model_input": args.model,
        "model_resolved": resolved_model,
        "is_openpangu_model": _is_openpangu_model(args.model),
        "tokenizer_name_or_path": getattr(tok, "name_or_path", None),
        "vocab_size": int(getattr(tok, "vocab_size", -1)),
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "prompt": args.prompt,
        "generated_ids_len": int(gen_ids.numel()),
        "generated_ids_head": gen_ids[:128].tolist(),
        "decoded_skip_special_true": decoded_skip_true,
        "decoded_skip_special_false": decoded_skip_false,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
