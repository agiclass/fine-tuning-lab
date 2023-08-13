# 项目描述

本工程是配合agiclass进行FineTune 和 Lora课程的实验部分，内容包含：基于ChatGLM-6B和ChatGLM2-6B两个模型的训练脚本，数据是 chain-ai-law-chanllege/cail2019 数据集。



### 运行环境

基于https://autodl.com 平台，具体操作过程请树东补充。



### 模型准备

获取 ChatGLM-6B  和 ChatGLM2-6B 到本地



### 获取路径

假定路径是 ~/.offline/

比如作者的目录：

```
hello@ubuntu:~/.offline$ tree -L 2
.
└── THUDM
    ├── chatglm2-6b
    └── chatglm-6b
```


### 下载实验脚本 

git clone https://github.com/taliux/Student_Experiments.git 

[Note: 可以做一个镜像把这个放好，这样用户就不用下载了]

进入Student_Experiments 目录

```
.
├── chatglm              #用于chatglm-6b的训练的目录
├── chatglm2						 #用于chatglm2-6b的训练的目录
└── data								 #已经处理好的法律数据，原始数据在数据集部分介绍如何获取
```

### 训练Pt2和验证结果

####  基于ChatGLM-6B进行Pt2 训练

```
cd chatglm
vim train_pt2.sh  #确认一下chatglm模型文件的位置，假设是~/.offline/THUDM/chatglm-6b
```

则修改模型目录为：

```bash
#!/bin/bash
PRE_SEQ_LEN=128
LR=2e-2
timestamp=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 python3 main_pt2.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ../data/LawChat.lite/train.jsonl \
    --validation_file ../data/LawChat.lite/dev.jsonl \
    --test_file ../data/LawChat.lite/test.jsonl \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path "~/.offline/THUDM/chatglm-6b" \   # 修改这一行，指定你的模型位置
    --output_dir "output/chatglm-6b-pt/$timestamp" \
    --max_source_length 1024 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \ 
    --per_device_eval_batch_size 8 \ 
    --gradient_accumulation_steps 8 \ 
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 30 \
    --num_train_epochs 1 \ 
    --logging_steps 30 \
    --save_steps 30 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
```



修改之后保存即可启动
```
bash train_pt2.sh
```

如果没有错误就等待训练完即可





#### 检查Pt2结果

训练完成之后如何查看结果，使用 目录中的 web_demo_pt2.sh

```bash
PRE_SEQ_LEN=128

MODULE_PATH="PATH/TO/YOUR/MODULE"   # 修改为你的模型位置"~/.offline/THUDM/chatglm-6b"
CHECKPOINT_DIR="PATH/TO/YOUR/CHECKPOINT" # 修改为你生成好的checkpoint的目录

CUDA_VISIBLE_DEVICES=0 python3 web_demo_pt2.py \
    --model_name_or_path $MODULE_PATH \
    --ptuning_checkpoint $CHECKPOINT_DIR \
    --pre_seq_len $PRE_SEQ_LE
```



修改之后执行 
```
bash  web_demo_pt2.sh
```

如果看到如下界面，就说明启动完成
```
Loading checkpoint shards: 100%|█████████████████████████████| 7/7 [00:04<00:00,  1.68it/s]
Some weights of ChatGLMForConditionalGeneration were not initialized from the model checkpoint at /home/hello/.offline/THUDM/chatglm2-6b/ and are newly initialized: ['transformer.prefix_encoder.embedding.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Running on local URL:  http://0.0.0.0:7860
```

然后查看你的服务器的IP地址，然后通过IP地址访问 7860端口就可以了

比如: 

```
curl ips.is   # 返回你服务器的公网地址
```

 

然后将地址贴到浏览器  http://xxx.xxx.xxx.xxx:7860  即可



### 训练Lora和验证结果

#### 基于ChatGLM-6B进行Lora训练

```
cd chatglm
vim train_lora.sh  #确认一下chatglm模型文件的位置，假设是~/.offline/THUDM/chatglm-6b
```

则修改模型目录为：

```bash
#!/bin/bash
PRE_SEQ_LEN=128
LR=2e-2
timestamp=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 python3 main_pt2.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ../data/LawChat.lite/train.jsonl \
    --validation_file ../data/LawChat.lite/dev.jsonl \
    --test_file ../data/LawChat.lite/test.jsonl \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path "~/.offline/THUDM/chatglm-6b" \   # 修改这一行，指定你的模型位置
    --output_dir "output/chatglm-6b-pt/$timestamp" \
    --max_source_length 1024 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \ 
    --per_device_eval_batch_size 8 \ 
    --gradient_accumulation_steps 8 \ 
    --predict_with_generate \
    --evaluation_strategy steps \
    --eval_steps 30 \
    --num_train_epochs 1 \ 
    --logging_steps 30 \
    --save_steps 30 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4
```



修改之后保存即可启动

```
bash train_lora.sh
```

如果没有错误就等待训练完即可





#### 检查Lora结果

训练完成之后如何查看结果，使用 目录中的 web_demo_lora.sh

```bash
PRE_SEQ_LEN=128

MODULE_PATH="PATH/TO/YOUR/MODULE"   # 修改为你的模型位置"~/.offline/THUDM/chatglm-6b"
CHECKPOINT_DIR="PATH/TO/YOUR/CHECKPOINT" # 修改为你生成好的checkpoint的目录

CUDA_VISIBLE_DEVICES=0 python3 web_demo_pt2.py \
    --model_name_or_path $MODULE_PATH \
    --ptuning_checkpoint $CHECKPOINT_DIR \
    --pre_seq_len $PRE_SEQ_LE
```



修改之后执行 

```
bash  web_demo_lora.sh
```

如果看到如下界面，就说明启动完成

```
Loading checkpoint shards: 100%|█████████████████████████████| 7/7 [00:04<00:00,  1.68it/s]
Some weights of ChatGLMForConditionalGeneration were not initialized from the model checkpoint at /home/hello/.offline/THUDM/chatglm2-6b/ and are newly initialized: ['transformer.prefix_encoder.embedding.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Running on local URL:  http://0.0.0.0:7860
```

然后查看你的服务器的IP地址，然后通过IP地址访问 7860端口就可以了

比如: 

```
curl ips.is   # 返回你服务器的公网地址	
```

 

然后将地址贴到浏览器  http://xxx.xxx.xxx.xxx:7860  即可

## 附录

### 数据集获取

```
wget https://github.com/china-ai-law-challenge/CAIL2019/raw/master/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3/data/big_train_data.json
wget https://github.com/china-ai-law-challenge/CAIL2019/raw/master/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3/data/dev_ground_truth.json
wget https://github.com/china-ai-law-challenge/CAIL2019/raw/master/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3/data/test_ground_truth.json

```