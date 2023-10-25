PRE_SEQ_LEN=128

DATA_FS="/root/autodl-tmp" # chatglm-6b所在目录
CHECKPOINT_DIR="/root/workspace/checkpoints/chatglm2-6b-pt2"

CUDA_VISIBLE_DEVICES=0 python3 web_demo.py \
    --model_name_or_path "${DATA_FS}/chatglm2-6b" \
    --ptuning_checkpoint $CHECKPOINT_DIR \
    --overwrite_cache \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
