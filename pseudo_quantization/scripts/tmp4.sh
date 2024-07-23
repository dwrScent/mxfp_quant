# CUDA_VISIBLE_DEVICES=4,5 ./scripts/llama_run_wiki.sh 13 wikitext 0 giant int 64 4 4
# CUDA_VISIBLE_DEVICES=4,5 ./scripts/llama_run_wiki.sh 30 wikitext 0 giant int 64 4 4
# CUDA_VISIBLE_DEVICES=4,5 ./scripts/llama2_run_wiki.sh 13 wikitext 0 giant int 64 4 4



# # giant-int
# CUDA_VISIBLE_DEVICES=4,5 ./scripts/llama_run_wiki.sh 13 wikitext 0 int int 64 4 4
# CUDA_VISIBLE_DEVICES=4,5 ./scripts/llama_run_wiki.sh 30 wikitext 0 int int 64 4 4
# CUDA_VISIBLE_DEVICES=4,5 ./scripts/llama2_run_wiki.sh 13 wikitext 0 int int 64 4 4


CUDA_VISIBLE_DEVICES=6,7 ./scripts/llama_run_wiki.sh 7 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=6,7 ./scripts/llama2_run_wiki.sh 7 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=6,7 ./scripts/llama_run_wiki.sh 13 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=6,7 ./scripts/llama2_run_wiki.sh 13 wikitext 0 giant int 64 4 4 1