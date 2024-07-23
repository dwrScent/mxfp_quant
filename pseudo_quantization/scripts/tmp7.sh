CUDA_VISIBLE_DEVICES=7 ./scripts/llama_run_wiki_dump.sh 13 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=7 ./scripts/llama2_run_wiki_load.sh 7 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=7 ./scripts/llama2_run_wiki_dump.sh 13 wikitext 0 giant int 64 4 4 1