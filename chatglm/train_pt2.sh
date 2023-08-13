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
    --model_name_or_path "/root/autodl-tmp/.offline/THUDM/chatglm-6b" \
    --output_dir "output/chatglm-6b-pt/$timestamp" \
    --max_source_length 1024 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 30 \
    --num_train_epochs 1 \
    --logging_steps 30 \
    --save_steps 30 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
