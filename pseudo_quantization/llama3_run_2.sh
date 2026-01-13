#!/bin/bash

TASKS=${1:-"wikitext"}
WBIT=4
WMODE="nvfp"
ABIT=16
AMODE="nvfp"
GROUP_SIZE=16
AWQ=False

# MODEL=meta-llama/Meta-Llama-3-8B
MODEL=/cephfs/shared/model/llama-3-8b-hf

python -m mxq.entry \
    --model_path "$MODEL" \
    --tasks "$TASKS" \
    --w_bit "$WBIT" \
    --w_mode "$WMODE" \
    --a_bit "$ABIT" \
    --a_mode "$AMODE" \
    --group_size "$GROUP_SIZE" --awq
