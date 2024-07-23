
Fork from https://github.com/mit-han-lab/llm-awq

## Setup

```shell

conda create -n awq python=3.10 -y
conda activate awq
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# CUDA 11.7，推荐使用 torch=2.0.1；没有测试过 torc=2.1.1

# 构建 kmeans kernel
cd awq/kmeans_kernel
python setup.py install

pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Usage

Evaluate LLaMa on multiple tasks with ANT data type (simulated pseudo quantization). Now we only use flint_0 (fp4, e2m1) in meta_flint set. You can select more data type in 4-bit meta_flint. 

```bash
# ANT W4A4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot -1 4 4
# ANT W8A8
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot -1 8 81
# 运行 65B 模型
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 65 wikitext 0 ant int-flint-float-pot -1 4 4
# 运行 OPT 模型
CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 ant int-flint-float-pot -1 4 4
# 测试 c4 数据集
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 c4 0 ant int-flint-float-pot -1 4 4


# OliVe
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 olive int-flint -1 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 olive int-flint -1 8 8

# Giant Ours
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 giant int 64 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/opt_run.sh 6.7 wikitext 0 giant int 64 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/bloom_run.sh 7 wikitext 0 giant int 64 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 giant int 64 4 16 0 5

# Giant-INT
CUDA_VISIBLE_DEVICES=0 ./scripts/llama_run_wiki.sh 7 wikitext 0 int int 64 4 4

# ANT, OliVe group-wise
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot 64 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot 128 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 ant int-flint-float-pot 128 8 8

CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 olive int 128 4 4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run_wiki.sh 7 wikitext 0 olive int 128 8 8
# Change the model_path based on your path.
```


## Setting

quant_mode = [giant, ant, olive, mokey, gobo, mx, awq]
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
