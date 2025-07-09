CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+2+max-a-sub_group+2+max quant e1m2
CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+4+max-a-sub_group+4+max quant e1m2
# OA
CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+2+outlier-a-sub_group+2+outlier quant e1m2
CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+4+outlier-a-sub_group+4+outlier quant e1m2

# Adaptive
CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+2+max-a-sub_group+2+max quant e1m2
CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+4+max-a-sub_group+4+max quant e1m2

CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+2+max-a-sub_group+2+outlier quant e1m2
CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+4+max-a-sub_group+4+outlier quant e1m2

# MX+
# CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+1+max-a-sub_group+1+max quant e0m3
# CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+1+outlier-a-sub_group+1+outlier quant e0m3
# CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+1+max-a-sub_group+1+max quant e0m3
# CUDA_VISIBLE_DEVICES=0,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+1+max-a-sub_group+1+outlier quant e0m3
