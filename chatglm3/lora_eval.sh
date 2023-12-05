#! /usr/bin/env bash
# MODEL_DIR="/root/autodl-tmp/chatglm3-6b"
# CHECKPOINT_DIR="/root/autodl-tmp/chatglm3-6b"
MODEL_DIR="/home/centos/models/chatglm3-6b"
CHECKPOINT_DIR="/home/centos/checkpoints/hotel_lora-chatglm3"

CUDA_VISIBLE_DEVICES=0 python cli_evaluate.py \
  --model $MODEL_DIR \
  --ckpt $CHECKPOINT_DIR \
  --data ../data/test.chatglm3.jsonl
