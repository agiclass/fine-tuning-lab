MODEL_PATH="/root/chatglm2-6b"
CHECKPOINT_DIR="PATH/TO/LORA/CHECKPOINT"

CUDA_VISIBLE_DEVICES=0 python3 web_demo.py \
    --model_name_or_path $MODEL_PATH \
    --lora_checkpoint $CHECKPOINT_DIR \
    --lora_rank 8 \
    --lora_alpha 32