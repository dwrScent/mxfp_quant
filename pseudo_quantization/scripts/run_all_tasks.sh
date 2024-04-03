

CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama_run_wiki.sh 7 wikitext 0 weighted_kmeans -1 4 output
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama_run.sh 7 arc_easy,hellaswag,piqa,winogrande,truthfulqa_mc 0 weighted_kmeans -1 4
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama_run.sh 7 arc_challenge 25 weighted_kmeans -1 4 output
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama_run.sh 7 hellaswag 10 weighted_kmeans -1 4 output
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama_run.sh 7 mmlu 5 weighted_kmeans -1 4 output

CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama_run_wiki.sh 7 wikitext 0 int-flint-float-pot -1 4 output
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama_run_wiki.sh 7 wikitext 0 nf -1 4 output

# ANT
CUDA_VISIBLE_DEVICES=3 ./scripts/llama_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot -1 4
CUDA_VISIBLE_DEVICES=2 ./scripts/opt_run.sh 6.7 wikitext 0 ant int-flint-float-pot -1 4

CUDA_VISIBLE_DEVICES=4 ./scripts/opt_run.sh 6.7 wikitext 0 ant int-flint-float-pot -1 4

CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama_run_wiki.sh 65 wikitext 0 ant int-flint-float-pot -1 4

# OliVe
CUDA_VISIBLE_DEVICES=3 ./scripts/llama_run_wiki.sh 7 wikitext 0 olive int-flint -1 4 bias=5,7

# CODE-ANT
CUDA_VISIBLE_DEVICES=3 ./scripts/llama_run_wiki.sh 7 wikitext 0 codeant int 64 4
