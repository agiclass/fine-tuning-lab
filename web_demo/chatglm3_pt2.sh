#!/bin/bash
MODEL_DIR="/root/autodl-tmp/chatglm3-6b"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints/hotel_pt2-chatglm3"

CUDA_VISIBLE_DEVICES=0 python webui_chatglm3.py \
  --model $MODEL_DIR \
  --ckpt $CHECKPOINT_DIR
