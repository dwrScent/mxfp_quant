
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/llama_run_wiki.sh 65 wikitext 0 olive int-flint 64 4 4

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/llama2_run_wiki.sh 70 wikitext 0 ant int-flint-float-pot 64 4 4
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/llama2_run_wiki.sh 70 wikitext 0 olive int-flint 64 4 4


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/llama_run_wiki_dump.sh 65 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/llama_run_wiki_dump.sh 30 wikitext 0 giant int 64 4 4 1