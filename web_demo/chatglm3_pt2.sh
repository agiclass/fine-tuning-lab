#!/bin/bash
MODEL_DIR="/root/autodl-tmp/chatglm3-6b"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints/chatglm3-6b-pt2"

CUDA_VISIBLE_DEVICES=0 python webui_chatglm3.py \
  --model_path $MODEL_DIR \
  --ckpt_path $CHECKPOINT_DIR
