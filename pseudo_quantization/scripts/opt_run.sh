MODEL_SIZE=${1:-"7"}
TASKS=${2:-"arc_challenge"}
SHOTS=${3:-"0"}
QUANT_MODE=${4:-"ant"}
ANT_MODE=${5:-"int"}
GROUP_SIZE=${6:-"-1"}
WEIGHT_BIT=${7:-"4"}
# MSE_TYPE=${8:-"weight"}
# OUTLIER_TYPE=${7:-"none"}
# OUTLIER_RATIO=${8:-"-1"}
DESC=${8:-""}

MODEL=/localdata_ssd/model/opt-${MODEL_SIZE}b
OUTPUT_NAME=opt-${MODEL_SIZE}b
OUTPUT_DIR=output/output_opt

mkdir -p $OUTPUT_DIR

python -m awq.entry_wikitext --model_path $MODEL \
    --tasks $TASKS \
    --num_fewshot $SHOTS \
    --w_bit $WEIGHT_BIT  \
    --q_backend fake \
    --no_zero_point \
    --quant_mode $QUANT_MODE \
    --ant_mode $ANT_MODE \
    --q_group_size $GROUP_SIZE \
    | tee $OUTPUT_DIR/${OUTPUT_NAME}_${TASKS}_${WEIGHT_BIT}bit_${SHOTS}shots_${QUANT_MODE}_${ANT_MODE}_g${GROUP_SIZE}_${DESC}.log 2>&1
