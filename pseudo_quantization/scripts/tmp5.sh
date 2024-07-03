CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 int int 128 4 16 0 43 1type-int

CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 16 0 43 4types
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 16 0 18 8types
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 16 0 8 18types
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 16 0 4 34types
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 16 0 2 66types
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 128 4 16 0 1 128types

CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 4 16 0 10
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 4 16 0 10 qkv
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 4 16 0 10 ffn
CUDA_VISIBLE_DEVICES=2 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 4 16 0 10 o
