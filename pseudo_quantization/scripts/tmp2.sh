CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run_wiki.sh 8 wikitext 0 mxfp int 32 w-1a4k16v16 w-base-a-base quant fp16
# CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run_wiki.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-base-a-base quant mxfp_base

CUDA_VISIBLE_DEVICES=3 ./scripts/llama3_run_wiki.sh 8 wikitext 0 ant float 32 w4a4k16v16 w-base-a-base quant mxfp_base

# CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run_wiki.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-dtype_search-a-dtype_search quant

# CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run_wiki.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-dtype_search-a-naive_adapt quant