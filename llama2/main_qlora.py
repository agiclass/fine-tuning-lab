import os
import sys
import torch
import logging
import argparse
import bitsandbytes as bnb
from functools import partial

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)

from trainer import QLoRATrainer
from arguments import ModelArguments, DataTrainingArguments, PeftArguments
from data_helper import load_raw_datasets, print_dataset_example
from data_preprocess import Preprocessor

from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM

logger = logging.getLogger(__name__)

def load_qlora(model,checkpoint_path, logger=None, merge=False):
    adapter_path = os.path.join(checkpoint_path, "adapter_model.bin")
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, checkpoint_path)
        if logger:
            logger.info(f"load checkpoint (adapter) from: {checkpoint_path}")
    else:
        sd_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(sd_path):
            model.load_state_dict(
                torch.load(sd_path), 
                strict=False
            )
            if logger:
                logger.info(f"load checkpoint (state_dict) from: {checkpoint_path}")
        else:
            if logger:
                logger.error(f"no checkpoint found at: {checkpoint_path}")
    
    if merge:
        model = model.merge_and_unload()
        
    return model

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{24500}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = '<pad>' #tokenizer.eos_token

    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def create_peft_config(modules,peft_args):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=peft_args.lora_rank,  # dimension of the updated matrices
        lora_alpha=peft_args.lora_alpha,  # parameter for scaling
        target_modules=modules,
        lora_dropout=peft_args.lora_dropout,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    # int4 如果调用PEFT原生的函数，计算的参数量会少一半

    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    logger.info(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

def main():

    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments, TrainingArguments))
    
    '''
    参数归类:
        model_args: Llama2模型自身的超参
        data_args: 数据集相关参数
        peft_args: 小参数量微调相关的超参
        training_args: 训练器相关参数
    '''
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_args.model_name_or_path, bnb_config)

    # 设置随机种子（以保证实验可复现）
    set_seed(training_args.seed)
    
    # Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # apply to Q K V matrix
    modules = ['q_proj', 'k_proj', 'v_proj']

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules,peft_args)

    raw_model = model
    model = get_peft_model(model, peft_config)

    # Load checkpoint if given
    if peft_args.lora_checkpoint is not None:
        model = load_qlora(
            raw_model,
            peft_args.lora_checkpoint,
            logger=logger,
            merge=False
        )


    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)
    
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
                data_processor.preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0],tokenizer)

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None, 
        padding=False
    )

    training_args.generation_max_length = data_args.max_source_length + data_args.max_target_length + 1
    training_args.generation_num_beams = 1
    
    trainer = QLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
        trainer.evaluate()

if __name__ == "__main__":
    main()
