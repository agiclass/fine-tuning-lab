#!/bin/bash
LORA_RANK=8
DATA_FS="/root/autodl-tmp"

CHECKPOINT_DIR="PATH/TO/YOUR/CHECKPOINT"

CUDA_VISIBLE_DEVICES=0 python3 main_qlora.py \
    --do_predict \
    --test_file ../data/test.jsonl \
    --prompt_column context \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path "${DATA_FS}/Llama-2-7b-hf" \
    --output_dir $CHECKPOINT_DIR \
    --lora_checkpoint $CHECKPOINT_DIR \
    --predict_with_generate \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --per_device_eval_batch_size 1 \
    --lora_rank $LORA_RANK \
    --lora_alpha 32 
