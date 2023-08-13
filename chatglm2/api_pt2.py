from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel,AutoConfig
import uvicorn, json, datetime
import torch
import os

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

from arguments import ModelArguments, DataTrainingArguments, PeftArguments

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, PeftArguments))
    model_args, peft_args = parser.parse_args_into_dataclasses()

    # 多显卡支持，使用下面三行代替上面两行，将num_gpus改为你实际的显卡数量
    # model_path = "THUDM/chatglm2-6b"
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = load_model_on_gpus(model_path, num_gpus=2)

    config = AutoConfig.from_pretrained( model_args.model_name_or_path, trust_remote_code=True)

    config.pre_seq_len = peft_args.pre_seq_len
    config.prefix_projection = peft_args.prefix_projection

    print(f"Loading prefix_encoder weight from {peft_args.ptuning_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(peft_args.ptuning_checkpoint, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    model = model.cuda()
    if peft_args.pre_seq_len is not None:
        # P-tuning v2
        model.transformer.prefix_encoder.float()
    
    model = model.eval()
    #model.eval()
    uvicorn.run(app, host='0.0.0.0', port=7860, workers=1)
