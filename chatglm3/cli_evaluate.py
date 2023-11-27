import os
import json
import torch
import argparse
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoConfig, AutoModel, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Evaluator:
    def __init__(self, tokenizer, model, data_path):
        self.tokenizer = tokenizer
        self.model = model
        self.data_path = data_path

    def _bleu4(self, pred, label):
        pred = pred.strip()
        label = label.strip()

        hypothesis = list(pred)
        reference = list(label)

        if len(hypothesis) == 0 or len(reference) == 0:
            return 0

        bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
        return bleu_score

    def _slot_accuracy(self, pred, label):
        correct = 0
        if pred is not None:
            for k, v in pred.items():
                if v is None:
                    continue
                if label and k in label:
                    if not isinstance(v,list):
                        correct += int(v==label[k])
                    else:
                        for t in v:
                            correct += int(t in label[k])

        pred_slots = sum(len(v) if isinstance(v, list) else 1 for v in pred.values()) if pred else 0
        true_slots = sum(len(v) if isinstance(v, list) else 1 for v in label.values()) if label else 0

        return correct, pred_slots, true_slots

    def _chat(self, query, history, role):
        eos_token_id = [self.tokenizer.eos_token_id, 
                        self.tokenizer.get_command("<|user|>"), 
                        self.tokenizer.get_command("<|observation|>")]
        inputs = self.tokenizer.build_chat_input(query, history=history, role=role)
        inputs = inputs.to('cuda')
        outputs = self.model.generate(**inputs, max_length=4096, eos_token_id=eos_token_id)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = self.tokenizer.decode(outputs)
        history.append({"role": role, "content": query})
        for response in response.split("<|assistant|>"):
            splited = response.split("\n", maxsplit=1)
            if len(splited) == 2:
                metadata, response = splited
            else:
                metadata = ""
                response = splited[0]

            if not metadata.strip():
                response = response.strip()
                history.append({"role": "assistant", "metadata": metadata, "content": response})
            else:
                history.append({"role": "assistant", "metadata": metadata, "content": response})
                response = "\n".join(response.split("\n")[1:-1])
                def tool_call(**kwargs):
                    return kwargs
                try:
                    parameters = eval(response)
                except:
                    parameters = {}
                response = {"name": metadata.strip(), "parameters": parameters}
        return response, history

    def evaluate(self):
        score_dict = {
            "slot_P": None,
            "slot_R": None,
            "slot_F1": None,
            "bleu-4": None,
        }
        bleu_scores = []
        true_slot_count = 0
        pred_slot_count = 0
        correct_slot_count = 0

        with open(self.data_path,'r') as f:
            test_data = [json.loads(line) for line in f]

        system_prompt = 'Answer the following questions as best as you can. You have access to the following tools:\n'

        for data in tqdm(test_data):
            dialog = data['conversations']
            tools_prompt = json.dumps(data['tools'],ensure_ascii=False)
            system_message = {'role': 'system', 'content': system_prompt+tools_prompt}
            history = [system_message]
            pred_slot, label_slot = {}, {}
            pred_reply, label_reply = "", ""
            for turn in dialog:
                if turn['role'] == 'user':
                    response, history = self._chat(turn['content'], history, 'user')
                    if isinstance(response, dict):
                        pred_slot = response['parameters']
                if turn['role'] == 'assistant':
                    if 'search_hotels' in turn['content']:
                        continue # skip assistant thought
                    else:
                        pred_reply = turn['content'].strip()
                        if pred_reply and label_reply:
                            score = self._bleu4(pred_reply, label_reply)
                            bleu_scores.append(score)
                            pred_reply, label_reply = "", ""
                if turn['role'] == 'tool':
                    label_slot = turn['parameters']
                    correct, pred_slots, true_slots = self._slot_accuracy(pred_slot, label_slot)
                    true_slot_count += true_slots
                    pred_slot_count += pred_slots
                    correct_slot_count += correct
                    pred_slot, label_slot = {}, {}
                    if 'observation' in turn:
                        response, history = self._chat(json.dumps(turn['observation'], ensure_ascii=False), history, 'observation')
                        if isinstance(response, str):
                            label_reply = response.strip()
        
        score_dict["slot_P"] = float(correct_slot_count/pred_slot_count) if pred_slot_count > 0 else 0
        score_dict["slot_R"] = float(correct_slot_count/true_slot_count) if true_slot_count > 0 else 0
        score_dict["slot_F1"] = 2*score_dict["slot_P"]*score_dict["slot_R"]/(score_dict["slot_P"]+score_dict["slot_R"]) if (score_dict["slot_P"]+score_dict["slot_R"]) > 0 else 0
        score_dict["bleu-4"] = sum(bleu_scores)/len(bleu_scores)
        print(f"score dict: {score_dict}")

def load_pt2(model_path, ckpt_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(ckpt_path, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model = model.to('cuda')
    return tokenizer, model

def load_lora(model_path, ckpt_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.half()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value"],
    )
    model = get_peft_model(model, peft_config)
    if os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin")):
        model.load_state_dict(torch.load(os.path.join(ckpt_path, "pytorch_model.bin")), strict=False)
    # model = model.merge_and_unload()
    model = model.to('cuda')
    return tokenizer, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, required=True, help="main model weights")
    parser.add_argument("--ckpt_path", type=str, default=None, required=True, help="The checkpoint path")
    args = parser.parse_args()

    tokenizer, model = load_pt2(args.model_path, args.ckpt_path)
    # tokenizer, model = load_lora(args.model_path, args.ckpt_path)

    evaluator = Evaluator(tokenizer, model, '../data/test.jsonl')
    evaluator.evaluate()
