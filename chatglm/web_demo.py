import os, sys

import gradio as gr
import mdtex2html

from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel
from arguments import ModelArguments, PeftArguments

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

def build_prompt(context,question,device):
    prompt = f"判例:\n{context}\n问题:\n{question}\n答案:\n"
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(device)
    return inputs

def generate(model, tokenizer, query: str, context: str, 
                max_length: int = 8192, 
                do_sample=False, 
                top_p=0.8, 
                temperature=0.8, 
                logits_processor=None,
                **kwargs
    ):
    
    if logits_processor is None:
            logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": max_length, "num_beams": 1, "do_sample": do_sample, "top_p": top_p,
                    "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    
    inputs = build_prompt(context,query,model.device)
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs)
    response = model.process_response(response)
    return response


model = None
tokenizer = None


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(context, input, max_length, top_p, temperature):
    
    response = generate(
            model, tokenizer, input, context, 
            max_length=max_length, top_p=top_p,
            temperature=temperature
    )

    return response


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return "", "", ""


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM-6B</h1>""")
    
    context = gr.Textbox(show_label=False, placeholder="判决书...", lines=10).style(
                    container=False)
    
    #chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="问题...", lines=5).style(
                    container=False)
            with gr.Column(scale=12):
                output = gr.Textbox(show_label=False, placeholder="答案...", lines=5).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    submitBtn.click(predict, [context, user_input, max_length, top_p, temperature],
                    [output], show_progress=True)

    emptyBtn.click(reset_state, outputs=[context, user_input, output], show_progress=True)


def main():
    global model, tokenizer

    parser = HfArgumentParser((ModelArguments, PeftArguments))
    model_args, peft_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)

    if peft_args.pre_seq_len is not None:
        config.pre_seq_len = peft_args.pre_seq_len
        config.prefix_projection = peft_args.prefix_projection

    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if peft_args.ptuning_checkpoint is not None:
        print(f"Loading prefix_encoder weight from {peft_args.ptuning_checkpoint}")
        prefix_state_dict = torch.load(os.path.join(peft_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    elif peft_args.lora_checkpoint is not None:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=peft_args.lora_rank,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=0,
            target_modules=["query_key_value"],
        )
        model = get_peft_model(model, peft_config)
        model.load_state_dict(torch.load(
                os.path.join(peft_args.lora_checkpoint, "pytorch_model.bin")
            ), 
            strict=False
        )
        #model = PeftModel.from_pretrained(model,peft_args.lora_checkpoint)
        #model = model.merge_and_unload()

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    
    model = model.cuda()
    model = model.half()
    
    if peft_args.pre_seq_len is not None:
        # P-tuning v2
        model.transformer.prefix_encoder.float()
    
    model = model.eval()
    demo.queue().launch(server_name="0.0.0.0", server_port=6006, share=False, inbrowser=True)



if __name__ == "__main__":
    main()