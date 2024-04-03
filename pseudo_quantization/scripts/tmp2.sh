# CUDA_VISIBLE_DEVICES=1,2 ./scripts/llama_run_wiki.sh 30 wikitext 0 olive int-flint -1 4 bias=5,7
CUDA_VISIBLE_DEVICES=1,2,3 ./scripts/llama_run_wiki.sh 65 wikitext 0 ant int-flint-float-pot -1 4
