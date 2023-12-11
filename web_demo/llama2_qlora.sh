MODEL_DIR="/root/autodl-tmp/Llama-2-7b-hf"
CHECKPOINT_DIR="/root/autodl-tmp/checkpoints/hotel_qlora-llama2"

CUDA_VISIBLE_DEVICES=0 python webui_llama2.py \
    --model_name_or_path $MODEL_DIR \
    --lora_checkpoint $CHECKPOINT_DIR \
    --overwrite_cache \
    --max_source_length 2048 \
    --max_target_length 1024 \
    --lora_rank 8 \
    --lora_alpha 32
