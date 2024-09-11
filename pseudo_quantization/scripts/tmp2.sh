CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama2_run.sh 70 arc_easy,hellaswag,piqa,winogrande 0 ant float 32 w-1a4k16v16 quant
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama2_run.sh 70 arc_easy,hellaswag,piqa,winogrande 0 ant float 32 w4a4k16v16 quant
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/llama2_run.sh 70 arc_easy,hellaswag,piqa,winogrande 0 mxfp int 32 w4a4k16v16 quant mxfp_base

