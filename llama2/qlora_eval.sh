#! /usr/bin/env bash
MODEL_DIR="/root/autodl-tmp/Llama-2-7b-hf"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints/hotel_qlora-llama2"

CUDA_VISIBLE_DEVICES=0 python cli_evaluate.py \
  --model $MODEL_DIR \
  --ckpt $CHECKPOINT_DIR \
  --data ../data/test.llama2.jsonl
