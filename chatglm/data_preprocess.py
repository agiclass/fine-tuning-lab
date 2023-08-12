from datasets import load_dataset

def print_dataset_example(example,tokenizer):
    print("input_ids",example["input_ids"])
    print("inputs", tokenizer.decode(example["input_ids"]))
    print("label_ids", example["labels"])
    print("labels", tokenizer.decode(example["labels"]))



def load_raw_datasets(data_args,cache_dir):
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    # 加载数据集
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=cache_dir
    )

    return raw_datasets

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
        测试数据的拼接方式：[pad][pad]...输入文本[gmask_token][bos_token][pad][pad]....输出文本[eos_token]
    '''
    def preprocess_function_eval(self,examples):  
        inputs, targets = [], []

        # 读取input/output即prompt/response字段的文本
        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                prompt = examples[self.prompt_column][i]
                inputs.append(prompt)
                targets.append(examples[self.response_column][i])

        self.tokenizer.truncation_side = 'left' #文本过长时从左侧截断（否则会把问题丢掉）
        
        # 对输入文本（prompt）做tokenize
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_source_length, 
            truncation=True, 
            padding=True
        )
        
        # 对输出文本（response）做tokenize
        labels = self.tokenizer(
            text_target=targets, 
            max_length=self.max_target_length, 
            truncation=True, 
            padding=True
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
        训练数据的拼接方式：输入文本[gmask_token][bos_token]输出文本[eos_token][pad][pad]....
    '''
    def preprocess_function_train(self,examples):
        max_seq_length = self.max_source_length + self.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                prompt, answer = examples[self.prompt_column][i], examples[self.response_column][i]

                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)

                # 手工做截断
                if len(a_ids) > self.max_source_length - 1: #留位置给special token
                    a_ids = a_ids[len(a_ids)-(self.max_source_length - 1):]

                # 手工做截断
                if len(b_ids) > self.max_target_length - 2: #留位置给special token
                    b_ids = b_ids[: self.max_target_length - 2]

                # 手工拼接
                input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = input_ids.index(self.tokenizer.bos_token_id) #输出开始的位置
                mask_position = context_length - 1

                # 输入部分标识为-100，计算loss时将被跳过
                labels = [-100] * context_length + input_ids[mask_position+1:]
                
                pad_len = max_seq_length - len(input_ids)

                # 手工pad
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [self.tokenizer.pad_token_id] * pad_len

                # 如果对pad token不进行loss计算，则将pad token标识为-100（模型约定的值）
                if self.ignore_pad_token_for_loss:
                    labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs