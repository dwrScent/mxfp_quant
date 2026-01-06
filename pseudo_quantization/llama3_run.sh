#!/bin/bash

TASKS=${1:-"wikitext"}
WBIT=4
WMODE="hif4"
ABIT=4
AMODE="hif4"
GROUP_SIZE=64

# MODEL=meta-llama/Meta-Llama-3-8B
MODEL=/cephfs/shared/model/llama-3-8b-hf

python -m mxq.entry \
    --model_path "$MODEL" \
    --tasks "$TASKS" \
    --w_bit "$WBIT" \
    --w_mode "$WMODE" \
    --a_bit "$ABIT" \
    --a_mode "$AMODE" \
    --group_size "$GROUP_SIZE"
