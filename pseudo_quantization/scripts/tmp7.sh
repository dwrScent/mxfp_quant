# CUDA_VISIBLE_DEVICES=7 ./scripts/llama_run_wiki_load.sh 7 wikitext 0 giant int 64 4 8 1
# CUDA_VISIBLE_DEVICES=7 ./scripts/llama_run_wiki_load.sh 13 wikitext 0 giant int 64 4 8 1
# CUDA_VISIBLE_DEVICES=7 ./scripts/llama2_run_wiki_load.sh 7 wikitext 0 giant int 64 4 8 1
# CUDA_VISIBLE_DEVICES=7 ./scripts/llama2_run_wiki_load.sh 13 wikitext 0 giant int 64 4 8 1


CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 ant int-flint-float-pot -1 4 4
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 olive int-flint -1 4 4
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 ant int-flint-float-pot -1 8 8
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 olive int-flint -1 8 8