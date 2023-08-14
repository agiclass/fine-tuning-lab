#!/bin/bash
LORA_RANK=8

CHECKPOINT_DIR="PATH/TO/YOUR/CHECKPOINT"

CUDA_VISIBLE_DEVICES=0 python3 main_lora.py \
    --do_predict \
    --test_file ../data/LawChat.lite/test.jsonl \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path "/root/chatglm2-6b" \
    --output_dir $CHECKPOINT_DIR \
    --lora_checkpoint $CHECKPOINT_DIR \
    --predict_with_generate \
    --max_source_length 1024 \
    --max_target_length 64 \
    --per_device_eval_batch_size 4 \
    --lora_rank $LORA_RANK \
    --lora_alpha 32 
