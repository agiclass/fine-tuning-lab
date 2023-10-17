#!/bin/bash
PRE_SEQ_LEN=128
LR=2e-2
#timestamp=$(date +%Y%m%d_%H%M%S)
DATA_FS="/root/autodl-tmp"

LOCAL_RANK=-1 CUDA_VISIBLE_DEVICES=0 python3 main_pt2.py \
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
    --output_dir "output/chatglm2-6b-pt" \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 320 \
    --num_train_epochs 1 \
    --logging_steps 320 \
    --save_steps 320 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 