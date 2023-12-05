#!/bin/bash
MODEL_DIR="/root/autodl-tmp/chatglm3-6b"

CUDA_VISIBLE_DEVICES=0 python webui_chatglm3.py \
  --model $MODEL_DIR
