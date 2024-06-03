CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 olive int-flint -1 4 bias=5,7
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 13 wikitext 0 olive int-flint -1 4 bias=5,7

CUDA_VISIBLE_DEVICES=3 ./scripts/llama_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot -1 4 verify


CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run.sh 7 cnn_dailymail 0 ant int-flint -1 -1

CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 ant int-flint -1 -1
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 codeant int -1 4

CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki_dump.sh 7 wikitext 0 codeant int 64 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki_load.sh 7 wikitext 0 codeant int 64 4

CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 codeant int 64 4

CUDA_VISIBLE_DEVICES=0 ./scripts/bloom_run.sh 7 wikitext 0 codeant int 64 4
