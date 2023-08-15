PRE_SEQ_LEN=128

MODEL_PATH="/root/chatglm2-6b"
CHECKPOINT_DIR="output/chatglm-6b-pt/checkpoint-600/"

CUDA_VISIBLE_DEVICES=0 python3 web_demo.py \
    --model_name_or_path $MODEL_PATH \
    --ptuning_checkpoint $CHECKPOINT_DIR \
    --pre_seq_len $PRE_SEQ_LEN 