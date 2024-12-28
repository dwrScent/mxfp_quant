# CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 giant int 64 w4a8k16v16 load

# CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 int int 64 w4a8k16v16 quant


# CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 mxfp int 32 w4a4k16v16 quant
# CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 mxfp int 32 w4a4k4v4 quant

# CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 ant int 32 w16a16k16v16 quant

# CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 ant float 32 w4a4k16v16 quant
# CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 ant float 64 w4a4k16v16 quant
# CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 ant float 128 w4a4k16v16 quant

# CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 quant e2m1
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-dtype_search-a-naive_adapt quant

CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 mxfp int 32 w16a16k16v16
CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 mxfp int 32 w16a16k16v16 w-dtype_search-a-naive_adapt quant
CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 mxfp int 32 w16a16k16v16 w-dtype_search-a-naive_adapt quant
CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 mxfp int 32 w16a16k16v16 w-base-a-base quant

CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 mxfp int 32 w4a4k16v16 w-sub_group_v2-a-sub_group_v2
# CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 mxfp int 32 w4a4k16v16 w-sub_group_v2-a-sub_group
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-sub_group_v2-a-sub_group_v2
# CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-sub_group_v2-a-sub_group


CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext,arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp int 32 w4a4k16v16 w-base-a-base
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext,arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp int 32 w4a4k16v16 w-base-a-base

CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 arc_challenge 0 mxfp int 32 w16a16k16v16 w-base-a-base
