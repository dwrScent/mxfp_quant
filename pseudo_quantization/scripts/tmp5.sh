
# CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-sub_group_v2-a-sub_group
# CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run.sh 8 arc_easy 0 mxfp int 32 w4a4k16v16 w-sub_group_v2-a-sub_group
# # CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run.sh 8 hellaswag 0 mxfp int 32 w4a4k16v16 w-sub_group_v2-a-sub_group
# CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run.sh 8 piqa 0 mxfp int 32 w4a4k16v16 w-sub_group_v2-a-sub_group
# CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run.sh 8 winogrande 0 mxfp int 32 w4a4k16v16 w-sub_group_v2-a-sub_group
# CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run.sh 8 arc_challenge 0 mxfp int 32 w4a4k16v16 w-sub_group_v2-a-sub_group

CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run.sh 8 piqa 0 mxfp int 32 w4a4k16v16 w-base-a-base
CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run.sh 8 winogrande 0 mxfp int 32 w4a4k16v16 w-base-a-base
CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run.sh 8 arc_challenge 0 mxfp int 32 w4a4k16v16 w-base-a-base