
Fork from https://github.com/mit-han-lab/llm-awq

## Setup

```shell

conda create -n mxq python=3.10 -y
conda activate mxq
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# CUDA 12.3，推荐使用 torch=2.3.1
pip install protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers_stream_generator -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## Usage

Before running the scripts, you need to change the $MODEL$ (in scripts) and the path of the calibration dataset (in `get_calib_dataset`).

Evaluate LLaMa on multiple tasks with MXFP.

```bash
# baseline MXFP4 
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
# group-wise data type search [E3M0, E2M1, E1M2], basic data type is MXINT
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp int 32 w4a4q16k16v16 w-dtype_search-a-dtype_search quant
# MX+, E0M3 for maximum value in group; sub_group_size=1, sub_group_mode=max
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+1+max-a-sub_group+1+max quant
# sub_group_size=4, sub_group_mode=max
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group+4+max-a-sub_group+4+max quant
# sub-group adaptive data types [E2M1, E1M2], default subgroup size is 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_adaptive+4+max-a-sub_group_adaptive+4+max quant

# baseline FP4-G32 with FP16 scale
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 ant float 32 w4a4q16k16v16 w-base-a-base quant mxfp_base

# oehter models or sizes
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama2_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
# zero-show tasks with MXFP4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 arc_easy,hellaswag,piqa,winogrande,arc_challenge,boolq 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
# backup tasks
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 lambada 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant

```

### Other baselines
```shell
# Run NVFP
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 nvfp float 16 w4a4q16k16v16 w-base-a-base quant
# SMXFP

# Reasoning Task, need to set `apply_chat_template` and `fewshot_as_multiturn` to True
CUDA_VISIBLE_DEVICES=1 ./scripts/qwen_run.sh 1.5 gsm8k_cot 8 mxfp float 32 w-1a16q16k16v16
CUDA_VISIBLE_DEVICES=1 ./scripts/qwen_run.sh 1.5 gpqa_diamond_cot_n_shot 8 mxfp float 32 w-1a16q16k16v16

CUDA_VISIBLE_DEVICES=1 ./scripts/qwen_run.sh 1.5 agieval_aqua_rat 4 mxfp float 32 w16a16q16k16v16
CUDA_VISIBLE_DEVICES=1 ./scripts/qwen_run.sh 1.5 asdiv 4 mxfp float 32 w-1a16q16k16v16
```

## Setting


## TODO list

+ [x] Add the sub-group size to the shell option, 2025-07-06 10:15:48 :white_check_mark:
+ [x] Divide the mode better (max-aware, outlier-aware, adaptive), 2025-07-06 10:15:44 :white_check_mark:
