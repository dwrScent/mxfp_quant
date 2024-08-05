CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama_run_wiki_load.sh 65 wikitext 0 giant int 64 4 4 0
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama_run_wiki_load.sh 65 wikitext 0 int int 64 4 4 0
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama_run_wiki_load.sh 65 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama_run_wiki_load.sh 65 wikitext 0 int int 64 4 4 1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama_run_wiki_load.sh 65 wikitext 0 giant int 64 4 8 1

CUDA_VISIBLE_DEVICES=4,5 ./scripts/llama_run_wiki_load.sh 30 wikitext 0 giant int 64 4 4 0
CUDA_VISIBLE_DEVICES=4,5 ./scripts/llama_run_wiki_load.sh 30 wikitext 0 int int 64 4 4 0
CUDA_VISIBLE_DEVICES=4,5 ./scripts/llama_run_wiki_load.sh 30 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=4,5 ./scripts/llama_run_wiki_load.sh 30 wikitext 0 int int 64 4 4 1
CUDA_VISIBLE_DEVICES=4,5 ./scripts/llama_run_wiki_load.sh 30 wikitext 0 giant int 64 4 8 1

CUDA_VISIBLE_DEVICES=6 ./scripts/llama_run_wiki_load.sh 7 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=6 ./scripts/llama_run_wiki_load.sh 13 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=6 ./scripts/llama2_run_wiki_load.sh 7 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=6 ./scripts/llama2_run_wiki_load.sh 13 wikitext 0 giant int 64 4 4 1

CUDA_VISIBLE_DEVICES=7 ./scripts/llama_run_wiki_load.sh 7 wikitext 0 giant int 64 4 8 1
CUDA_VISIBLE_DEVICES=7 ./scripts/llama_run_wiki_load.sh 13 wikitext 0 giant int 64 4 8 1
CUDA_VISIBLE_DEVICES=7 ./scripts/llama2_run_wiki_load.sh 7 wikitext 0 giant int 64 4 8 1
CUDA_VISIBLE_DEVICES=7 ./scripts/llama2_run_wiki_load.sh 13 wikitext 0 giant int 64 4 8 1

CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 int int 64 16 16 0

CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 ant int-flint-float-pot -1 4 4
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 olive int-flint -1 4 4
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 ant int-flint-float-pot -1 8 8
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 olive int-flint -1 8 8

CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 giant int 64 4 4 0
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 giant int 64 4 8 0
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 giant int 64 4 8 1

CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 int int 64 4 4 0
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 int int 64 4 8 0
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 int int 64 4 4 1
CUDA_VISIBLE_DEVICES=7 ./scripts/opt_run.sh 6.7 wikitext 0 int int 64 4 8 1
