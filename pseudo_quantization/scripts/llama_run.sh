MODEL_SIZE=${1:-"7"}
TASKS=${2:-"arc_challenge"}
SHOTS=${3:-"0"}
ANT_MODE=${4:-"int"}
GROUP_SIZE=${5:-"-1"}
WEIGHT_BIT=${6:-"4"}
MSE_TYPE=${7:-"weight"}
# OUTLIER_TYPE=${7:-"none"}
# OUTLIER_RATIO=${8:-"-1"}
DESC=${8:-""}

MODEL=/localdata_ssd/model/llama-${MODEL_SIZE}b-hf-transformers-4.29
OUTPUT_NAME=llama-${MODEL_SIZE}b
OUTPUT_DIR=output/output_nf_ant_wiki

mkdir -p $OUTPUT_DIR

python -m awq.entry --model_path $MODEL \
    --tasks $TASKS \
    --num_fewshot $SHOTS \
    --w_bit $WEIGHT_BIT  \
    --q_backend fake \
    --no_zero_point \
    --ant_mode $ANT_MODE \
    --mse_type $MSE_TYPE \
    --ant_asym 0 \
    --w_low 100 --w_high 105 \
    --q_group_size $GROUP_SIZE \
    | tee $OUTPUT_DIR/${OUTPUT_NAME}_${TASKS}_${WEIGHT_BIT}bit_${SHOTS}shots_${ANT_MODE}_g${GROUP_SIZE}_${MSE_TYPE}_${DESC}.log 2>&1
