#!/bin/bash

# TASKS=${1:-"GSM8K"}
TASKS=${1:-"SuperGPQA"}
QUANT_METHOD="mxfp"
BACKEND="transformers"
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MODEL_PARALLEL=${MODEL_PARALLEL:-0}
MAX_SAMPLES=${MAX_SAMPLES:-}

# MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL=FreedomIntelligence/openPangu-R-72B-2512

EXTRA_ARGS=()
if [ "$MODEL_PARALLEL" = "1" ]; then
  EXTRA_ARGS+=(--model_parallel)
fi
if [ -n "$MAX_SAMPLES" ]; then
  EXTRA_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python -m mxq.evaluation.inference \
        --model "$MODEL" \
        --dataset "$TASKS" \
        --backend "$BACKEND" \
        --quant_method $QUANT_METHOD \
        --max_model_length "$MAX_MODEL_LENGTH" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --trust_remote_code \
        "${EXTRA_ARGS[@]}"
