LORA_RANK=8
DATA_FS="/root/autodl-tmp" # Llama-2-7b-hf所在目录
CHECKPOINT_DIR="/root/workspace/checkpoints/llama2-7b-qlora"

CUDA_VISIBLE_DEVICES=0 python3 web_demo.py \
    --model_name_or_path "${DATA_FS}/Llama-2-7b-hf" \
    --lora_checkpoint $CHECKPOINT_DIR \
    --overwrite_cache \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --lora_rank $LORA_RANK \
    --lora_alpha 32 
