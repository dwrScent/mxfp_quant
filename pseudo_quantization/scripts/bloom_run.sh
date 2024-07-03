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

MODEL=/localssd/wmhu/models/bloom-7b1/snapshots/e83e90ba86f87f74aa2731cdab25ccf33976bd66
OUTPUT_NAME=bloom-7b1
OUTPUT_DIR=output/output_bloom

mkdir -p $OUTPUT_DIR

python -m awq.entry --model_path $MODEL \
    --tasks $TASKS \
    --num_fewshot $SHOTS \
    --w_bit $WEIGHT_BIT  \
    --a_bit $ACT_BIT  \
    --quant_kv $QUANT_KV \
    --a_stride $A_STRIDE \
    --q_backend fake \
    --no_zero_point \
    --quant_mode $QUANT_MODE \
    --ant_mode $ANT_MODE \
    --q_group_size $GROUP_SIZE \
    | tee $OUTPUT_DIR/${OUTPUT_NAME}_${TASKS}_w${WEIGHT_BIT}a${ACT_BIT}_kv${QUANT_KV}_${SHOTS}shots_${QUANT_MODE}_${ANT_MODE}_g${GROUP_SIZE}_astride${A_STRIDE}_${MSE_TYPE}_${DESC}_$(date +%m%d%H%M).log 2>&1
