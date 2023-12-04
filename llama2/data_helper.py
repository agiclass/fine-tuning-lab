from datasets import load_dataset
import numpy as np

def remove_minus100(ids,val):
    """
        -100是HF预留的id（不参与loss计算）
        有的tokenizer在decode -100时会报错
        因此在decode之前去除（替换为pad_id）
    """
    ids = np.array(ids)
    ids = np.where(ids == -100, val, ids)
    return ids

def print_dataset_example(example,tokenizer):
    print("input_ids",example["input_ids"])
    print("inputs", tokenizer.decode(example["input_ids"],skip_special_tokens=True))
    print("label_ids", example["labels"])
    label_ids = remove_minus100(example["labels"],tokenizer.pad_token_id)
    print("labels", tokenizer.decode(label_ids,skip_special_tokens=True))

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
