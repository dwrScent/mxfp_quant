# CUDA_VISIBLE_DEVICES=6 ./scripts/opt_run_dump.sh 6.7 wikitext 0 giant int 64 4 4 0
# CUDA_VISIBLE_DEVICES=6 ./scripts/opt_run_load.sh 6.7 wikitext 0 giant int 64 4 8 0
CUDA_VISIBLE_DEVICES=6 ./scripts/opt_run_load.sh 6.7 wikitext 0 giant int 64 4 4 1
CUDA_VISIBLE_DEVICES=6 ./scripts/opt_run_load.sh 6.7 wikitext 0 giant int 64 4 8 1

# CUDA_VISIBLE_DEVICES=6 ./scripts/opt_run.sh 6.7 wikitext 0 int int 64 4 4 0
# CUDA_VISIBLE_DEVICES=6 ./scripts/opt_run.sh 6.7 wikitext 0 int int 64 4 8 0
# CUDA_VISIBLE_DEVICES=6 ./scripts/opt_run.sh 6.7 wikitext 0 int int 64 4 4 1
# CUDA_VISIBLE_DEVICES=6 ./scripts/opt_run.sh 6.7 wikitext 0 int int 64 4 8 1