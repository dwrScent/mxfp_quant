CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 olive int-flint -1 4 bias=5,7
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 13 wikitext 0 olive int-flint -1 4 bias=5,7

CUDA_VISIBLE_DEVICES=3 ./scripts/llama_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot -1 4 verify