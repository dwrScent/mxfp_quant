CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant e1m2
# MX+
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group+1+max-a-sub_group+1+max quant e1m2
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group+2+max-a-sub_group+2+max quant e1m2
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group+4+max-a-sub_group+4+max quant e1m2
# OA
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group+1+outlier-a-sub_group+1+outlier quant e1m2
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group+2+outlier-a-sub_group+2+outlier quant e1m2
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group+4+outlier-a-sub_group+4+outlier quant e1m2
