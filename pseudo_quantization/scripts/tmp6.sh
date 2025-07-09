# CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
# CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
# CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
# CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
# CUDA_VISIBLE_DEVICES=0 ./scripts/mistral_run.sh 7 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant

# CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama2_run.sh 7 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant mxfp_keep_outlier
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama3_run.sh 8 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant mxfp_keep_outlier
# # CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant mxfp_keep_outlier
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/opt_run.sh 6.7 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant mxfp_keep_outlier
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/mistral_run.sh 7 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant mxfp_keep_outlier

CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+1+max-a-sub_group+1+max quant e0m3
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+1+outlier-a-sub_group+1+outlier quant e0m3
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+1+max-a-sub_group+1+max quant e0m3
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+1+max-a-sub_group+1+outlier quant e0m3

CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant mxfp_base

CUDA_VISIBLE_DEVICES=0,1,3 ./scripts/llama3_run.sh 70 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant mxfp_base

CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 nvfp float 32 w4a4q16k16v16 w-base-a-base quant