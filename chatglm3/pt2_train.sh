#! /usr/bin/env bash

set -ex

LR=2e-2
PRE_SEQ_LEN=128
MAX_SEQ_LEN=2048
MAX_STEP=1000

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=hotel_pt
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

BASE_MODEL_PATH=/root/autodl-tmp/chatglm3-6b

CUDA_VISIBLE_DEVICES=0 python main_pt2.py \
    --do_train \
    --train_file ../data/train.multi.jsonl \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --logging_dir $OUTPUT_DIR/logs \
    --save_steps 200 \
    --learning_rate $LR \
    --quantization_bit 8 \
    --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
