# Adaptive
CUDA_VISIBLE_DEVICES=3 ./scripts/llama2_run.sh 7 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+1+max-a-sub_group+1+max quant e1m2
CUDA_VISIBLE_DEVICES=3 ./scripts/llama2_run.sh 7 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+2+max-a-sub_group+2+max quant e1m2
CUDA_VISIBLE_DEVICES=3 ./scripts/llama2_run.sh 7 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+4+max-a-sub_group+4+max quant e1m2

CUDA_VISIBLE_DEVICES=3 ./scripts/llama2_run.sh 7 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+1+max-a-sub_group+1+outlier quant e1m2
CUDA_VISIBLE_DEVICES=3 ./scripts/llama2_run.sh 7 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+2+max-a-sub_group+2+outlier quant e1m2
CUDA_VISIBLE_DEVICES=3 ./scripts/llama2_run.sh 7 arc_easy,hellaswag,piqa,winogrande,arc_challenge 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+4+max-a-sub_group+4+outlier quant e1m2