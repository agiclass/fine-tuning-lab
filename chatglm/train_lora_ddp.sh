#!/bin/bash
LR=2e-3
LORA_RANK=8
timestamp=$(date +%Y%m%d_%H%M%S)

torchrun --nproc_per_node=8 --master_port=29501  main_lora.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ../data/LawChat.lite/train.jsonl \
    --validation_file ../data/LawChat.lite/dev.jsonl \
    --test_file ../data/LawChat.lite/test.jsonl \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path "../.offline/THUDM/chatglm-6b" \
    --output_dir "output/chatglm-6b-lora-ddp/$timestamp" \
    --max_source_length 1024 \
    --max_target_length 64 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --num_train_epochs 1 \
    --logging_steps 100 \
    --save_steps 100 \
    --learning_rate $LR \
    --lora_rank $LORA_RANK \
    --lora_alpha 32 \
    --lora_dropout 0.1

