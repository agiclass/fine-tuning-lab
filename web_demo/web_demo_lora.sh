#!/bin/bash
LORA_RANK=8
DATA_FS="/root/autodl-tmp" # chatglm-6b所在目录
CHECKPOINT_DIR="/root/workspace/checkpoints/chatglm2-6b-lora"

CUDA_VISIBLE_DEVICES=0 python3 web_demo.py \
    --model_name_or_path "${DATA_FS}/chatglm2-6b" \
    --lora_checkpoint $CHECKPOINT_DIR \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --lora_rank $LORA_RANK \
    --lora_alpha 32 
