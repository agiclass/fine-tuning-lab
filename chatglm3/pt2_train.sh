#! /usr/bin/env bash

set -ex

PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
MAX_SEQ_LEN=2048
GRAD_ACCUMULARION_STEPS=16
MAX_STEP=1000
SAVE_INTERVAL=500

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=hotel_pt
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

# BASE_MODEL_PATH=/autodl-tmp/chatglm3-6b
BASE_MODEL_PATH=/home/tong.liu/chatglm3-6b

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS main_pt2.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ../data/train.jsonl \
    --validation_file ../data/dev.jsonl \
    --test_file ../data/test.jsonl \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --overwrite_cache \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --logging_steps 1 \
    --logging_dir $OUTPUT_DIR/logs \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --quantization_bit 8 \
    --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
