
Fork from https://github.com/mit-han-lab/llm-awq

## Setup

```shell

conda create -n mxq python=3.10 -y
conda activate mxq
pip install --upgrade pip  # enable PEP 660 support
pip install vllm==0.7.0 --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .

cd mxq/eval
pip install -e .

# CUDA 12.3，推荐使用 torch=2.3.1, lm-eval==0.4.4 (Also works for latest lm-eval==0.4.9)
pip install protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers_stream_generator -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## Usage

Before running the scripts, you need to change the $MODEL$ (in scripts) and the path of the calibration dataset (in `get_calib_dataset`).

Evaluate LLaMa on multiple tasks with MXFP.

```bash
# baseline MXFP4 
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
# NVFP, g16
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 nvfp float 16 w4a4q16k16v16 w-base-a-base quant
# SMX
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 smxfp float 16 w4a4q16k16v16 w-base-a-base quant

# MX-ANT, MX-olive, MX-MANT, MicroscopiQ
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxant float 32 w4a4q16k16v16 w-sub_group_adaptive+32+max-a-sub_group_adaptive+32+max
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 olive int 32 w4a4q16k16v16
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxmant float 32 w4a4q16k16v16 w-sub_group_adaptive+32+max-a-sub_group_adaptive+32+max
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mcq float 32 w4a4q16k16v16

# Ours, M2XFP
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp float 32 w4a4q16k16v16 w-sub_group_es+8+max-a-sub_group_em_real+8+max quant


# zero-show tasks with MXFP4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 arc_easy,hellaswag,piqa,winogrande,arc_challenge,boolq 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant
# run AIME-2025 on DeepSeek-R1-Distill-Qwen-1.5B
CUDA_VISIBLE_DEVICES=0 ./scripts/deepseek_run.sh 1.5 AIME-2025 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant

```


### Old setting

```shell
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

# run AIME-2025 on DeepSeek-R1-Distill-Qwen-1.5B
CUDA_VISIBLE_DEVICES=0 ./scripts/deepseek_run.sh 1.5 AIME-2025 0 mxfp float 32 w4a4q16k16v16 w-base-a-base quant

```


## Setting


## TODO list

+ [x] Add the sub-group size to the shell option, 2025-07-06 10:15:48 :white_check_mark:
+ [x] Divide the mode better (max-aware, outlier-aware, adaptive), 2025-07-06 10:15:44 :white_check_mark:
