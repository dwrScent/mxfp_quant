CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot 64 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 13 wikitext 0 ant int-flint-float-pot 64 4 4

CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot 64 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 13 wikitext 0 ant int-flint-float-pot 64 4 4

CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 ant int-flint-float-pot 64 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/bloom_run.sh 7 wikitext 0 ant int-flint-float-pot 64 4 4

CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 olive int-flint 64 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 13 wikitext 0 olive int-flint 64 4 4


CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 olive int-flint 64 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 13 wikitext 0 olive int-flint 64 4 4

CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 olive int-flint 64 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/bloom_run.sh 7 wikitext 0 olive int-flint 64 4 4

CUDA_VISIBLE_DEVICES=2,3,4,5 ./scripts/llama_run_wiki.sh 30 wikitext 0 ant int-flint-float-pot 64 4 4
CUDA_VISIBLE_DEVICES=2,3,4,5 ./scripts/llama_run_wiki.sh 30 wikitext 0 olive int-flint 64 4 4


CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot 64 4 4 fusion

CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 int int 64 4 4

CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki_load.sh 7 wikitext 0 giant int 64 4 4 1 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_load.sh 7 coqa 0 giant int 64 4 4 1 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_load.sh 7 gsm8k 0 giant int 64 4 4 1 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_load.sh 7 truthfulqa_gen 0 giant int 64 4 4 1 10