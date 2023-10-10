from datasets import load_dataset
import numpy as np

def print_dataset_example(example,tokenizer):
    print("input_ids",example["input_ids"])
    print("inputs", tokenizer.decode(example["input_ids"],skip_special_tokens=True))
    print("label_ids", example["labels"])
    label_ids = example["labels"]
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    print("labels", tokenizer.decode(example["labels"],skip_special_tokens=True))

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