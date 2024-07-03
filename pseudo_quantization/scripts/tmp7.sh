CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 4 16 0 10
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama2_run_wiki.sh 13 wikitext 0 giant int 64 4 16 0 10

CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 16 0 10
CUDA_VISIBLE_DEVICES=2,3 ./scripts/llama2_run_wiki.sh 13 wikitext 0 giant int 128 4 16 0 10


CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 16 16 1 10

# CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 8 1 10 16types
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 4 8 1 43 4types
CUDA_VISIBLE_DEVICES=1 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 4 8 1 18 8types
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 4 8 1 4 34types
CUDA_VISIBLE_DEVICES=3 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 4 8 1 2 66types

CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 8 1 43 4types
CUDA_VISIBLE_DEVICES=1 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 8 1 18 8types
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 8 1 4 34types
CUDA_VISIBLE_DEVICES=3 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 8 1 2 66types
CUDA_VISIBLE_DEVICES=3 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 8 1 1 66types

CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 32 4 8 1 43 4types
CUDA_VISIBLE_DEVICES=1 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 32 4 8 1 18 8types
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 32 4 8 1 4 34types
CUDA_VISIBLE_DEVICES=3 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 32 4 8 1 1 66types

CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 256 4 8 1 43 4types
CUDA_VISIBLE_DEVICES=1 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 256 4 8 1 18 8types
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 256 4 8 1 4 34types
CUDA_VISIBLE_DEVICES=3 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 256 4 8 1 1 66types