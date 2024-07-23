# CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama2_run_wiki.sh 7 wikitext 0 int int 64 4 4 1
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 13 wikitext 0 int int 64 4 4 1

CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 int int 64 4 4 1
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 13 wikitext 0 int int 64 4 4 1
# CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 30 wikitext 0 int int 64 4 4 1

# debug for 2 device
CUDA_VISIBLE_DEVICES=6,7 ./scripts/llama_run_wiki.sh 7 wikitext 0 int int 64 4 4 1


# CUDA_VISIBLE_DEVICES=6,7 ./scripts/llama_run_wiki_dump.sh 7 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama_run_wiki_dump.sh 30 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama_run_wiki_dump.sh 65 wikitext 0 giant int 64 4 4 1


CUDA_VISIBLE_DEVICES=7 ./scripts/llama_run_wiki_dump.sh 13 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=7 ./scripts/llama2_run_wiki_load.sh 7 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=7 ./scripts/llama2_run_wiki_dump.sh 13 wikitext 0 giant int 64 4 4 1

