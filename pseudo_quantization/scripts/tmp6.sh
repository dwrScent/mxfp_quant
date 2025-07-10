CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+1+max-a-sub_group+1+max quant e0m3
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+1+outlier-a-sub_group+1+outlier quant e0m3
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+1+max-a-sub_group+1+max quant e0m3
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+1+max-a-sub_group+1+outlier quant e0m3