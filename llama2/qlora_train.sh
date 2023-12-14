#! /usr/bin/env bash

set -ex

LR=2e-4
LORA_RANK=8

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=hotel_qlora
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}
mkdir -p $OUTPUT_DIR

DATA_FS="/root/autodl-tmp"

LOCAL_RANK=-1 CUDA_VISIBLE_DEVICES=0 python main_qlora.py \
    --do_train \
    --do_eval \
    --train_file ../data/train.llama2.jsonl \
    --validation_file ../data/dev.llama2.jsonl \
    --prompt_column context \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path "${DATA_FS}/Llama-2-7b-hf" \
    --output_dir $OUTPUT_DIR \
    --optim "paged_adamw_8bit" \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --logging_dir $OUTPUT_DIR/logs \
    --save_steps 160 \
    --learning_rate $LR \
    --lora_rank $LORA_RANK \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --fp16 \
    --warmup_ratio 0.1 \
    --seed 23 2>&1 | tee ${OUTPUT_DIR}/train.log
