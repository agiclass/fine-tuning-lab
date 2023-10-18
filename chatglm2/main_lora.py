#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')
sys.path.append('.')
import logging
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

from data_preprocess import Preprocessor
from common.trainer_seq2seq import Seq2SeqTrainer
from common.arguments import ModelArguments, DataTrainingArguments, PeftArguments
from common.data_helper import load_raw_datasets, print_dataset_example
from common.evaluator import Evaluator, save_predictions
from common.checkpoint_helper import load_lora_checkpoint

from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel

logger = logging.getLogger(__name__)

def load_model(model_name):
    # 加载ChatGLM的Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 加载模型
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    model = model.half()
    model.is_parallelizable = True
    model.model_parallel = True
    return model, tokenizer

def create_peft_config(peft_args):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=peft_args.lora_rank,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        target_modules=["query_key_value"],
    )
    return peft_config

def setup_logger(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def main():

    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments, Seq2SeqTrainingArguments))
    
    '''
    参数归类:
        model_args: ChatGLM模型自身的超参
        data_args: 数据集相关参数
        peft_args: 小参数量微调相关的超参
        training_args: 训练器相关参数
    '''
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    setup_logger(training_args)
    
    logger.warning(f"Training/evaluation parameters {training_args}")


    # 设置随机种子（以保证实验可复现）
    set_seed(training_args.seed)

    # 加载模型和Tokenizer
    model, tokenizer = load_model(model_args.model_name_or_path)
    
    # 生成PEFT配置
    peft_config = create_peft_config(peft_args)

    raw_model = model

    model = get_peft_model(model, peft_config).cuda()

    if peft_args.lora_checkpoint is not None:
        model = load_lora_checkpoint(
            raw_model,
            peft_args.lora_checkpoint,
            logger=logger,
            merge=not training_args.do_train
        )
        model = model.cuda()


    # 加载数据集
    raw_datasets = load_raw_datasets(data_args,model_args.cache_dir)

    data_processor = Preprocessor(
        data_args=data_args,
        tokenizer=tokenizer
    )

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        # 随机排序训练集
        train_dataset = raw_datasets["train"].shuffle(training_args.seed)
        
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                data_processor.preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        print_dataset_example(train_dataset[0],tokenizer)

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                data_processor.preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0],tokenizer)

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                data_processor.preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(predict_dataset[0],tokenizer)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None, 
        padding=False
    )

    # Metric
    evaluator = Evaluator(tokenizer)

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = data_args.max_source_length + data_args.max_target_length + 1
    training_args.generation_num_beams = 1


    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=evaluator.compute_metrics, # 训练过程中是否阶段性跑测试（否则直接计算loss）
        save_lora=True #是否只保存训练的参数
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        logger.info(f"checkpoints save to: {training_args.output_dir}")

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model(os.path.join(training_args.output_dir,"checkpoint-last"))

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=False, max_length=training_args.generation_max_length)
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    results = {}
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_new_tokens=data_args.max_target_length, num_beams=training_args.generation_num_beams, do_sample=False)
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(predict_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        save_predictions(predict_results,tokenizer,training_args.output_dir)


    return results


if __name__ == "__main__":
    main()
