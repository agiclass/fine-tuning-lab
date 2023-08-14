#!/bin/bash
PRE_SEQ_LEN=128
LR=2e-2
timestamp=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 python3 main_pt2.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ../data/LawChat.lite/train.jsonl \
    --validation_file ../data/LawChat.lite/dev.jsonl \
    --test_file ../data/LawChat.lite/test.jsonl \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path "/root/autodl-tmp/chatglm2-6b" \
    --output_dir "output/chatglm2-6b-pt/$timestamp" \
    --max_source_length 1024 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 60 \
    --num_train_epochs 1 \
    --logging_steps 60 \
    --save_steps 60 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4