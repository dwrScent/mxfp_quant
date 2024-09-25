

CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki_load.sh 7 wikitext 0 giant int 64 4 4 1 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_load.sh 7 coqa 0 giant int 64 4 4 1 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_load.sh 7 gsm8k 0 giant int 64 4 4 1 10

CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_load.sh 7 truthfulqa_gen 0 giant int 64 4 4 1 
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_load.sh 7 truthfulqa_gen 0 giant int 64 4 8 0


CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 w4a8k16v16 load



CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 arc_easy,hellaswag,piqa,winogrande 0 ant int -1 w-1a4k16v16 quant
# float g32
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 arc_easy,hellaswag,piqa,winogrande 0 ant float 32 w4a4k16v16 quant
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 arc_easy,hellaswag,piqa,winogrande 0 mxfp int 32 w4a4k16v16 quant

CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama2_run.sh 70 arc_easy,hellaswag,piqa,winogrande 0 mxfp int 32 w4a4k16v16 w-base-a-base quant

CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama3_run_wiki.sh 70 wikitext 0 ant float 32 w4a4k16v16 w-base-a-base quant

CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run_wiki.sh 8 wikitext 0 mxfp int 32 w-1a4k16v16 w-base-a-base quant fp16
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run_wiki.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-base-a-base quant mxfp_base
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run_wiki.sh 8 wikitext 0 ant float 32 w4a4k16v16 w-base-a-base quant mxfp_base

CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run_wiki.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-dtype_search-a-dtype_search quant

CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run_wiki.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-dtype_search-a-naive_adapt quant
