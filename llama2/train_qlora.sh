#!/bin/bash
LR=2e-4
LORA_RANK=8
#timestamp=$(date +%Y%m%d_%H%M%S)

DATA_FS="/root/autodl-tmp"

LOCAL_RANK=-1 CUDA_VISIBLE_DEVICES=0 python3 main_qlora.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ../data/train.jsonl \
    --validation_file ../data/dev.jsonl \
    --test_file ../data/test.jsonl \
    --prompt_column context \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path "${DATA_FS}/Llama-2-7b-hf" \
    --output_dir "output/llama2-7b-qlora" \
    --optim "paged_adamw_8bit" \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 160 \
    --num_train_epochs 1 \
    --logging_steps 160 \
    --save_steps 160 \
    --learning_rate $LR \
    --lora_rank $LORA_RANK \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --fp16 \
    --warmup_ratio 0.1 \
    --seed 23

