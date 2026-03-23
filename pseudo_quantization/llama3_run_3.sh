#!/bin/bash

TASKS=${1:-"wikitext"}
# TASKS=${1:-"boolq"}
WBIT=4
WMODE="nvesm2"
ABIT=4
AMODE="nvesm2"
GROUP_SIZE=16

# MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=/cephfs/shared/model/llama-3-8b-hf
# MODEL=/cephfs/shared/model/llama-2-7b-hf
# MODEL=/cephfs/shared/model/DeepSeek-R1-Distill-Qwen-7B
MODEL=/cephfs/shared/model/Qwen-7B

python -m mxq.entry \
    --model_path "$MODEL" \
    --tasks "$TASKS" \
    --w_bit "$WBIT" \
    --w_mode "$WMODE" \
    --a_bit "$ABIT" \
    --a_mode "$AMODE" \
    --group_size "$GROUP_SIZE"
