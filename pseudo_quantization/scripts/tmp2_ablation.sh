
# group ablation
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 codeant int 32 4 16 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 codeant int 64 4 16 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 codeant int 128 4 16 0 10
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 codeant int 256 4 16 0 10

# data type ablation

# 8 types 0,17,25,50,75,100,125,int
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 codeant int 64 4 16 0 25
# 16 types
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 codeant int 64 4 16 0 10

# 32 左右
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 codeant int 64 4 16 0 4
