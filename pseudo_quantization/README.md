
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
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp float 32 w4a4k16v16 w-base-a-base quant
# group-wise data type search [E3M0, E2M1, E1M2]
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-dtype_search-a-dtype_search quant
# MX+, E0M3 for maximum value in group
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-sub_group-a-sub_group quant
# sub-group adaptive data types [E2M1, E1M2], default subgroup size is 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 mxfp int 32 w4a4k16v16 w-sub_group_v2-a-sub_group_v2 quant


# baseline FP4-G32 with FP16 scale
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 wikitext 0 ant float 32 w4a4k16v16 w-base-a-base quant mxfp_base

# oehter models or sizes
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama2_run.sh 70 wikitext 0 mxfp float 32 w4a4k16v16 w-base-a-base quant
CUDA_VISIBLE_DEVICES=0,1 ./scripts/llama3_run.sh 70 wikitext 0 mxfp float 32 w4a4k16v16 w-base-a-base quant
CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 mxfp float 32 w4a4k16v16 w-base-a-base quant
# zero-show tasks with MXFP4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama3_run.sh 8 arc_easy,hellaswag,piqa,winogrande 0 mxfp float 32 w4a4k16v16 w-base-a-base quant
```

## Setting

quant_mode = [giant, ant, olive, mokey, gobo, mxfp, awq]
+ giant: W4A8, W4KV4A8
+ ant: W4A4, W8A8. ant do not quantize the attention, and target CNN and BERT.
+ olive: W4A4, W8A8. olive do not quantize the attention
+ mokey: W4A4. Mokey only evaluate the BERT model
+ GOBO: W8A16. GOBO can not quantize the KV
+ AWQ: W4A16. weight-only

## TODO list

+ [x] add compute encode gen and compute encode mode
+ [ ] ~~algorithm / method to select the data type~~
+ [ ] add KV quantization; 8 data type for kV
+ [x] add ANT and OliVe
+ [ ] ~~For KV, select data type through variance~~
