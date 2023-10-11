#!/bin/bash
PRE_SEQ_LEN=128
DATA_FS="/root/autodl-tmp"
CHECKPOINT_DIR="PATH/TO/YOUR/CHECKPOINT"

CUDA_VISIBLE_DEVICES=0 python3 main_pt2.py \
    --do_predict \
    --test_file ../data/test.jsonl \
    --prompt_column context \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path "${DATA_FS}/chatglm2-6b" \
    --output_dir $CHECKPOINT_DIR \
    --ptuning_checkpoint $CHECKPOINT_DIR \
    --predict_with_generate \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --per_device_eval_batch_size 8 \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

