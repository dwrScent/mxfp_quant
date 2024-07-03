# ANT
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot -1 4
CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 ant int-flint-float-pot -1 4

CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 ant int-flint-float-pot -1 4

CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama_run_wiki.sh 65 wikitext 0 ant int-flint-float-pot -1 4

# OliVe
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 olive int-flint -1 4 bias=5,7

# CODE-ANT
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 giant int 64 4
