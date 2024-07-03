CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 giant int 64 4 4 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 13 wikitext 0 giant int 64 4 4 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 30 wikitext 0 giant int 64 4 4 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 65 wikitext 0 giant int 64 4 4 0 10

CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 4 4 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 13 wikitext 0 giant int 64 4 4 0 10

CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 giant int 64 4 4 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/bloom_run.sh 7 wikitext 0 giant int 64 4 4 0 10


CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 giant int 64 8 8 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 13 wikitext 0 giant int 64 8 8 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 30 wikitext 0 giant int 64 8 8 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 65 wikitext 0 giant int 64 8 8 0 10

CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 8 8 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 13 wikitext 0 giant int 64 8 8 0 10

CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 giant int 64 8 8 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/bloom_run.sh 7 wikitext 0 giant int 64 8 8 0 10