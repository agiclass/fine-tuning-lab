# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import typer
import functools
import dataclasses as dc
from pathlib import Path
import ruamel.yaml as yaml
from typing import Annotated, Optional, Union

from peft import (
    PeftConfig,
    get_peft_config,
    get_peft_model
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments, 
)

from trainer import PrefixTrainer, LoRATrainer
from preprocess import MultiTurnDataset

app = typer.Typer(pretty_exceptions_show_locals=False)

def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser


@dc.dataclass
class FinetuningConfig(object):
    model_path: str
    train_file: str
    val_file: str
    max_seq_length: int
    quantization_bit: Optional[int] = None

    training_args: Seq2SeqTrainingArguments = dc.field(
        default=Seq2SeqTrainingArguments(output_dir='./output')
    )
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.val_file is None:
            # skips the evaluation stage when `do_eval` or `eval_file` is not provided
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
            self.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                self.training_args.per_device_eval_batch_size
                or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = _resolve_path(path)
        kwargs = _get_yaml_parser().load(path)
        return cls.from_dict(**kwargs)


def print_model_size(model: PreTrainedModel):
    print("--> Model")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6}M params\n")


# TODO: Not sure if this is necessary, can set it to half
def _prepare_model_for_training(model: torch.nn.Module):
    for param in model.parameters():
        if param.requires_grad:
	    # if train with cpu, cast all params to fp32 instead of trainable ones.
            param.data = param.data.to(torch.float32)


def load_tokenizer_and_model(
        model_dir: str,
        ft_config: Optional[PeftConfig] = None,
) -> tuple[PreTrainedTokenizer, torch.nn.Module]:
    peft_config = ft_config.peft_config
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if peft_config is not None:
        if peft_config.peft_type.name == "PREFIX_TUNING":
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            config.pre_seq_len = peft_config.num_virtual_tokens
            config.use_cache = False
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                config=config,
            )
        if peft_config.peft_type.name == "LORA":
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                empty_init=False,
                use_cache=False
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False
        )
    if ft_config.quantization_bit is not None:
        model = model.quantize(ft_config.quantization_bit)
    print_model_size(model)
    return tokenizer, model


@app.command()
def main(config_file: Annotated[str, typer.Argument(help='')]):
    ft_config = FinetuningConfig.from_file(config_file)
    tokenizer, model = load_tokenizer_and_model(ft_config.model_path, ft_config=ft_config)
    _prepare_model_for_training(model)

    if ft_config.training_args.do_train:
        with open(ft_config.train_file, "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]
        train_dataset = MultiTurnDataset(
            train_data,
            tokenizer,
            ft_config.max_seq_length,
        )

    if ft_config.training_args.do_eval:
        with open(ft_config.val_file, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f]
        eval_dataset = MultiTurnDataset(
            eval_data,
            tokenizer,
            ft_config.max_seq_length,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    if ft_config.peft_config.peft_type.name == "PREFIX_TUNING":
        trainer = PrefixTrainer(
            model=model,
            args=ft_config.training_args,
            train_dataset=train_dataset if ft_config.training_args.do_train else None,
            eval_dataset=eval_dataset if ft_config.training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    elif ft_config.peft_config.peft_type.name == "LORA":
        trainer = LoRATrainer(
            model=model,
            args=ft_config.training_args,
            train_dataset=train_dataset if ft_config.training_args.do_train else None,
            eval_dataset=eval_dataset if ft_config.training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    if ft_config.training_args.do_train:
        checkpoint = None
        if ft_config.training_args.resume_from_checkpoint is not None:
            checkpoint = ft_config.training_args.resume_from_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()

    if ft_config.training_args.do_eval:
        trainer.evaluate()

if __name__ == '__main__':
    app()
