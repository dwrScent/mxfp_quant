#!/bin/bash

task_name=${1:-"mnli"}
gpu_num=${2:-"1"}
SHOTS=${3:-"0"}
WEIGHT_BIT=${4:-"4"}
QUANT_MODE=${5:-"ant"}
ANT_MODE=${6:-"int"}
GROUP_SIZE=${7:-"-1"}

batch_size=64
path="ModelTC/bert-base-uncased-$task_name"

output_dir="./log/bert_ptq/$task_name"

mkdir -p $output_dir

export CUDA_VISIBLE_DEVICES=$gpu_num
python -m awq.entry_bert_mnli \
  --do_eval \
  --model_name_or_path $path \
  --task_name $task_name \
  --max_length 128 \
  --quantize_batch_size $batch_size \
  --per_device_eval_batch_size $batch_size \
  --output_dir ./log/bert_${size}_ptq/$task_name/ \
  --num_fewshot $SHOTS \
  --w_bit $WEIGHT_BIT  \
  --no_zero_point \
  --quant_mode $QUANT_MODE \
  --ant_mode $ANT_MODE \
  --q_group_size $GROUP_SIZE \
  | tee $output_dir/${OUTPUT_NAME}_${task_name}_${WEIGHT_BIT}bit_${SHOTS}shots_${QUANT_MODE}_${ANT_MODE}_g${GROUP_SIZE}.log 2>&1
  # > ./log/bert_${size}_ptq/$task_name/${batch_size}.out