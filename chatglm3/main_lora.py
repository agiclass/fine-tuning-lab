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
from trainer import PrefixTrainer
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
    model = get_peft_model(model, peft_config)
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

        # if training_args.local_rank < 1:
        #     sanity_check(train_dataset[0]['input_ids'], train_dataset[0]['labels'], tokenizer)

    if training_args.do_eval:
        with open(data_args.validation_file, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f]

        eval_dataset = MultiTurnDataset(
            eval_data,
            tokenizer,
            data_args.max_seq_length,
        )

        # if training_args.local_rank < 1:
        #     sanity_check(eval_dataset[0]['input_ids'], eval_dataset[0]['labels'], tokenizer)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    evaluator = Evaluator(tokenizer)

    # Initialize our Trainer
    trainer = PrefixTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=evaluator.compute_metrics if training_args.predict_with_generate else None,
        save_changed=True
    )

    if training_args.do_train:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=False, max_length=training_args.generation_max_length)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
