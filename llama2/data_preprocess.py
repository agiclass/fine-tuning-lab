import sys
import json
from prompt_helper import build_prompt, build_response

class Preprocessor:
    
    def __init__(self,data_args,tokenizer):
        self.prompt_column = data_args.prompt_column
        self.response_column = data_args.response_column
        self.max_source_length = data_args.max_source_length
        self.max_target_length = data_args.max_target_length
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = data_args.ignore_pad_token_for_loss
    
    # 处理测试(dev/test)数据
    '''
        测试数据的拼接方式：[pad][pad]...[bos_token]输入文本[pad][pad]....输出文本
    '''
    def preprocess_function_eval(self,examples):  
        inputs, targets = [], []

        # 读取input/output即prompt/response字段的文本
        inputs, targets = [], []
        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                context = examples[self.prompt_column][i]
                prompt = build_prompt(context)
                response = build_response( examples[self.response_column][i] )
                inputs.append(prompt)
                targets.append(response)

        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = 'left'

        # 对输入文本（prompt）做tokenize
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_source_length, 
            truncation=True, 
            padding=True
        )

        self.tokenizer.padding_side = 'right'

        # 对输出文本（response）做tokenize
        labels = self.tokenizer(
            text_target=targets, 
            max_length=self.max_target_length, 
            truncation=True, 
            padding=True,
            add_special_tokens=False
        )

        # 如果对pad token不进行loss计算，则将pad token标识为-100（模型约定的值）
        if self.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


    # 处理训练(train)数据
    '''
        训练数据的拼接方式：[bos_token]输入文本输出文本[eos_token][pad][pad]....
    '''
    def preprocess_function_train(self,examples):
        max_seq_length = self.max_source_length + self.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                context, response = examples[self.prompt_column][i], examples[self.response_column][i]
                prompt = build_prompt(context)
                response = build_response(response)

                #prompt = self.tokenizer.build_prompt(query)
                a_ids = self.tokenizer.encode(
                    text=prompt, 
                    add_special_tokens=False, 
                    truncation=True,
                    max_length=self.max_source_length-1
                )
                b_ids = self.tokenizer.encode(
                    text=response, 
                    add_special_tokens=False, 
                    truncation=True,
                    max_length=self.max_target_length-1
                )

			
                context_length = len(a_ids) + 1

                # 手工拼接
                input_ids = [self.tokenizer.bos_token_id] + a_ids + b_ids + [self.tokenizer.eos_token_id]
                
                # 手工pad
                labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]
                
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [self.tokenizer.pad_token_id] * pad_len

                # 如果对pad token不进行loss计算，则将pad token标识为-100（模型约定的值）
                if self.ignore_pad_token_for_loss:
                    labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]	

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)


        return model_inputs
