#!/bin/bash
MODEL_SIZE=${1:-"7"}
TASKS=${2:-"arc_challenge"}
SHOTS=${3:-"0"}
QUANT_MODE=${4:-"ant"}
ANT_MODE=${5:-"int"}
GROUP_SIZE=${6:-"-1"}
QUANT_BIT_WIDTH=${7:-"w4a8k16v16"}
MXFP_MODE=${8:-"w-base-a-base"}
OPTION=${9:-"quant"}
TOPK=${10:-"1"}
EM_BIT=${11:-"2"}
ES_BIT=${12:-"2"}

# MODEL=/cephfs/shared/model/llama-${MODEL_SIZE}b-hf-transformers-4.29
MODEL=/localssd/wmhu/model/llama-${MODEL_SIZE}b-hf-transformers-4.29
OUTPUT_NAME=llama-${MODEL_SIZE}b
if [ "$OPTION" == "load" ]; then
    OUTPUT_DIR=output/output_giant_load_$(date +%m%d)
    EXTRA_OPTION="--load_quant /localssd/wmhu/models/quant_cache/$OUTPUT_NAME-w4-g$GROUP_SIZE-$QUANT_MODE"
elif [ "$OPTION" == "dump" ]; then
    OUTPUT_DIR=output/output_giant_dump_$(date +%m%d)
    mkdir -p quant_cache
    EXTRA_OPTION="--dump_quant quant_cache/$OUTPUT_NAME-w4-g$GROUP_SIZE-$QUANT_MODE"
elif [ "$OPTION" == "quant" ]; then
    OUTPUT_DIR=output/output_llama2_hpca_motivation_$(date +%m%d)
    EXTRA_OPTION=""
else
    echo "Invalid option: $OPTION. Only 'load', 'dump', and 'quant' are supported."
    exit 1
fi

mkdir -p $OUTPUT_DIR

python -m mxq.entry --model_path $MODEL \
    --tasks $TASKS \
    --num_fewshot $SHOTS \
    --quant_bit_width $QUANT_BIT_WIDTH \
    $EXTRA_OPTION \
    --q_backend fake \
    --no_zero_point \
    --quant_mode $QUANT_MODE \
    --mxfp $MXFP_MODE \
    --ant_mode $ANT_MODE \
    --q_group_size $GROUP_SIZE \
    --topk $TOPK \
    --em_bit $EM_BIT \
    --es_bit $ES_BIT \
    | tee $OUTPUT_DIR/${OUTPUT_NAME}_${TASKS}_${QUANT_BIT_WIDTH}_${SHOTS}shots_${QUANT_MODE}_${ANT_MODE}_g${GROUP_SIZE}_${MXFP_MODE}_${TOPK}_${EM_BIT}_${ES_BIT}_$(date +%m%d%H%M).log 2>&1