python run_simulator.py \
  --models llama3_8b \
  --accelerators olive,ant,mant,nvesm2 \
  --normalized-bench olive \
  --batch-size 1

# python run_simulator.py \
#   --models llama3_8b \
#   --accelerators olive,ant,mant,microscopiq,m2xfp,nvesm2,nvfp \
#   --normalized-bench olive \
#   --batch-size 1

# Full results in the paper
# python run_simulator.py \
#   --models llama2_7b,llama3_8b,falcon_7b,mistral_7b,opt6b7,llama3_70b \
#   --accelerators olive,ant,mant,microscopiq,m2xfp,nvesm2,nvfp \
#   --normalized-bench olive \
#   --batch-size 1 \
#   2>& 1| tee results/run.log
