CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 int int 64 4 4 1 5 kv4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 13 wikitext 0 int int 64 4 4 1 5 kv4

CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 int int 64 4 4 1 5 kv4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 13 wikitext 0 int int 64 4 4 1 5 kv4

CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 30 wikitext 0 int int 64 4 4 1 5 kv4
