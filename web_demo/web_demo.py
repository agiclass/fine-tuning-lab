#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')
import os
import re
import json
import torch
import numpy as np
import gradio as gr
import pandas as pd
from transformers import AutoModel, AutoTokenizer, HfArgumentParser
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from db_client import HotelDB
from common.checkpoint_helper import load_lora_checkpoint
from common.prompt_helper import build_prompt, build_response
from common.arguments import ModelArguments, DataTrainingArguments, PeftArguments
from common.evaluator import parse_json

def init_model():
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments))
    model_args, data_args, peft_args = parser.parse_args_into_dataclasses()
    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # 加载Model
    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True).half()
    model.is_parallelizable = True
    model.model_parallel = True
    # 加载LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=peft_args.lora_rank,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        target_modules=["query_key_value"],
    )
    raw_model = model
    model = get_peft_model(model, peft_config).cuda()
    if peft_args.lora_checkpoint is not None:
        model = load_lora_checkpoint(raw_model, peft_args.lora_checkpoint).cuda()
    return model, tokenizer, data_args.max_source_length, data_args.max_target_length

def get_completion(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_source_length)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_target_length + max_source_length + 1, num_beams=1, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    return response

# init gloab variables
db = HotelDB()
model, tokenizer, max_source_length, max_target_length = init_model()

def chat(user_input, chatbot, context, search_field, return_field):
    context.append({'role':'user','content':user_input})
    response = get_completion(build_prompt(context))
    # 判断以search命令开头时去执行搜索
    if response.strip().startswith("search:"):
        # 取出最新一条 'search:' 后面的json查询条件
        search_query = parse_json(response)
        if search_query is not None:
            search_field = json.dumps(search_query,indent=4,ensure_ascii=False)
            context.append({'role':'search','arguments':search_field})

            # 调用酒店查询接口
            return_field = db.search(search_field, limit=3)
            context.append({'role':'return','records':return_field})
            keys = return_field[0].keys() if return_field else []
            data = {key: [item[key] for item in return_field] for key in keys}
            data = data or {"hotel": []}
            return_field = pd.DataFrame(data)
        
            # 将查询结果发给LLM，再次那么让LLM生成回复
            response = get_completion(build_prompt(context))
    
    
    start = response.find(":")+1
    reply = response[start:].strip()
    chatbot.append((user_input, reply))
    return "", chatbot, context, search_field, return_field

def reset_state():
    return [], []

def main():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">Hotel Chatbot</h1>""")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
                user_input = gr.Textbox(show_label=False, placeholder="输入框...", lines=2)
                with gr.Row():
                    submitBtn = gr.Button("提交", variant="primary")
                    emptyBtn = gr.Button("清空")
            with gr.Column(scale=2):
                gr.HTML("""<h4>Search</h4>""")
                search_field = gr.Textbox(show_label=False, placeholder="search...")
                gr.HTML("""<h4>Return</h4>""")
                return_field = gr.Dataframe()

        context = gr.State([])

        submitBtn.click(chat, [user_input, chatbot, context, search_field, return_field],
                        [user_input, chatbot, context, search_field, return_field])
        emptyBtn.click(reset_state, outputs=[chatbot, context])

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=6006, inbrowser=True)


if __name__ == "__main__":
    main()
