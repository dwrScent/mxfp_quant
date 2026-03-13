#!/bin/bash

# TASKS=${1:-"GSM8K"}
TASKS=${1:-"SuperGPQA"}
QUANT_METHOD="mxfp"
BACKEND="transformers"

# MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL=FreedomIntelligence/openPangu-R-72B-2512

python -m mxq.evaluation.inference \
        --model "$MODEL" \
        --dataset "$TASKS" \
        --backend "$BACKEND" \
        --quant_method $QUANT_METHOD \
        --trust_remote_code
