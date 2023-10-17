#!/bin/bash
LR=2e-3
LORA_RANK=8
#timestamp=$(date +%Y%m%d_%H%M%S)
DATA_FS="/root/autodl-tmp"

LOCAL_RANK=-1 CUDA_VISIBLE_DEVICES=0 python3 main_lora.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ../data/train.jsonl \
    --validation_file ../data/dev.jsonl \
    --test_file ../data/test.jsonl \
    --prompt_column context \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path "${DATA_FS}/chatglm2-6b" \
    --output_dir "output/chatglm2-6b-lora" \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 20 \
    --num_train_epochs 1 \
    --logging_steps 20 \
    --save_steps 20 \
    --learning_rate $LR \
    --lora_rank $LORA_RANK \
    --lora_alpha 32 \
    --lora_dropout 0.1 

