
MODEL_PATH="/root/chatglm2-6b"

CUDA_VISIBLE_DEVICES=0 python3 web_demo.py \
    --model_name_or_path $MODEL_PATH 

