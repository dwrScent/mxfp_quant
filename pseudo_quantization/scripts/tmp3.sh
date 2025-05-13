CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama2_run.sh 70 wikitext 0 mxfp float 32 w4a16k16v16 w-base-a-base quant mxfp4-baseline
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama2_run.sh 70 wikitext 0 ant float 32 w4a16k16v16 w-base-a-base quant fp4
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama2_run.sh 70 wikitext 0 mxfp float 32 w4a16k16v16 w-sub_group-a-base quant mx+