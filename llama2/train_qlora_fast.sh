#!/bin/bash
LR=2e-4
LORA_RANK=8
#timestamp=$(date +%Y%m%d_%H%M%S)

DATA_FS="/root/autodl-tmp"

LOCAL_RANK=-1 CUDA_VISIBLE_DEVICES=0 python3 main_qlora_fast.py \
    --do_train \
    --do_eval \
    --train_file ../data/train.jsonl \
    --validation_file ../data/dev.jsonl \
    --prompt_column context \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path "${DATA_FS}/Llama-2-7b-hf" \
    --output_dir "output/llama2-7b-qlora" \
    --optim "paged_adamw_8bit" \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy steps \
    --eval_steps 40 \
    --num_train_epochs 1 \
    --logging_steps 40 \
    --save_steps 40 \
    --learning_rate $LR \
    --lora_rank $LORA_RANK \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --fp16 \
    --load_best_model_at_end  

