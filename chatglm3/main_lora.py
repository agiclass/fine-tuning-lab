import logging
import sys
import json
import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from trainer import LoRATrainer
from evaluator import Evaluator
from arguments import ModelArguments, DataTrainingArguments, PeftArguments
from preprocess_utils import sanity_check, MultiTurnDataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

logger = logging.getLogger(__name__)

def setup_logger(training_args):
    ###### Setup logging ######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

def load_lora_model(model_args, peft_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = model.half()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=peft_args.lora_rank,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        target_modules=["query_key_value"],
    )
    model = get_peft_model(model, peft_config).to("cuda")
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False
    return tokenizer, model

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments, Seq2SeqTrainingArguments))
    model_args, data_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    setup_logger(training_args)
    set_seed(training_args.seed)
    tokenizer, model = load_lora_model(model_args, peft_args)
    model.print_trainable_parameters()

    if training_args.do_train:
        with open(data_args.train_file, "r", encoding="utf-8") as f:
            train_data = [json.loads(line) for line in f]

        train_dataset = MultiTurnDataset(
            train_data,
            tokenizer,
            data_args.max_seq_length,
        )

        if training_args.local_rank < 1:
            sanity_check(train_dataset[0]['input_ids'], train_dataset[0]['labels'], tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    evaluator = Evaluator(tokenizer)

    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()

if __name__ == "__main__":
    main()
