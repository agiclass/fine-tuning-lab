PRE_SEQ_LEN=128

MODULE_PATH="PATH/TO/YOUR/MODULE"
CHECKPOINT_DIR="PATH/TO/YOUR/CHECKPOINT"

CUDA_VISIBLE_DEVICES=0 python3 web_demo_pt2.py \
    --model_name_or_path $MODULE_PATH \
    --ptuning_checkpoint $CHECKPOINT_DIR \
    --pre_seq_len $PRE_SEQ_LEN

