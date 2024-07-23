MODEL_SIZE=${1:-"7"}
TASKS=${2:-"arc_challenge"}
SHOTS=${3:-"0"}
QUANT_MODE=${4:-"ant"}
ANT_MODE=${5:-"int"}
GROUP_SIZE=${6:-"-1"}
WEIGHT_BIT=${7:-"4"}
ACT_BIT=${8:-"16"}
QUANT_KV=${9:-"0"}
A_STRIDE=${10:-"5"}
# MSE_TYPE=${11:-"weight"}
# OUTLIER_TYPE=${7:-"none"}
# OUTLIER_RATIO=${8:-"-1"}
DESC=${11:-""}

# MODEL=/localssd/wmhu/models/llama-${MODEL_SIZE}b-hf-transformers-4.29
MODEL=/localssd/wmhu/models//llama-2-${MODEL_SIZE}b-hf
OUTPUT_NAME=llama-2-${MODEL_SIZE}b
OUTPUT_DIR=output/output_llama2_kvtest

mkdir -p $OUTPUT_DIR
# --dump_quant $OUTPUT_NAME \
# --load_awq /localdata_ssd/wmhu/llm/llm-awq/awq-model-zoo/llama-7b-w4-g128.pt

python -m awq.entry --model_path $MODEL \
    --tasks $TASKS \
    --num_fewshot $SHOTS \
    --w_bit $WEIGHT_BIT  \
    --a_bit $ACT_BIT  \
    --quant_kv $QUANT_KV \
    --a_stride $A_STRIDE \
    --load_quant quant_cache/$OUTPUT_NAME-w$WEIGHT_BIT-g$GROUP_SIZE-$QUANT_MODE \
    --q_backend fake \
    --no_zero_point \
    --quant_mode $QUANT_MODE \
    --ant_mode $ANT_MODE \
    --q_group_size $GROUP_SIZE \
    | tee $OUTPUT_DIR/${OUTPUT_NAME}_${TASKS}_w${WEIGHT_BIT}a${ACT_BIT}_kv${QUANT_KV}_${SHOTS}shots_${QUANT_MODE}_${ANT_MODE}_g${GROUP_SIZE}_astride${A_STRIDE}_${MSE_TYPE}_${DESC}_$(date +%m%d%H%M).log 2>&1