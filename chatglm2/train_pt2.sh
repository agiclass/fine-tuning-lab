#!/bin/bash
PRE_SEQ_LEN=128
LR=2e-2
#timestamp=$(date +%Y%m%d_%H%M%S)

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
    --model_name_or_path "/root/chatglm2-6b" \
    --output_dir "output/chatglm2-6b-pt" \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 40 \
    --num_train_epochs 1 \
    --logging_steps 40 \
    --save_steps 40 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4