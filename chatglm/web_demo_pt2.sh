PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=0 python3 web_demo.py \
    --model_name_or_path THUDM/chatglm-6b \
    --ptuning_checkpoint LawChat/result/checkpoint-3000/ \
    --pre_seq_len $PRE_SEQ_LEN

