import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from prompt_helper import build_prompt
from main_qlora import load_model, load_qlora, create_bnb_config

def parse_json(string):
    search_pos = 0
    # 开始寻找第一个 '{'
    start = string.find('{', search_pos)
    if start == -1:
        return None
    # 从找到的 '{' 位置开始，向后寻找最后一个 '}'
    end = string.rfind('}', start)
    if end == -1:
        return None
    # 提取并尝试解析 JSON
    json_string = string[start:end + 1]
    try:
        obj = json.loads(json_string)
        return obj
    except json.JSONDecodeError:
        return None

class Evaluator:
    def __init__(self,tokenizer,model,data_path):
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

    def compute_metrics(self):
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

        # 读取测试集
        with open(self.data_path,'r') as f:
            test_data = [json.loads(line) for line in f]

        for data in tqdm(test_data):
            context = data['context']
            label = json.loads(data['response'])
            # 读取context并生成prompt填入模型
            prompt = build_prompt(context)
            inputs = self.tokenizer(prompt, return_token_type_ids=False, return_tensors="pt", truncation=True, padding=True, max_length=2048)
            inputs = inputs.to(model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=3073, num_beams=1, do_sample=False)
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            # 拿到模型输出文本进行解析，与label对比计算指标
            if label['role'] == 'search':
                if 'search:' in response:
                    pred = parse_json(response)
                else:
                    pred = {}
                correct, pred_slots, true_slots = self._slot_accuracy(pred, label['arguments'])
                true_slot_count += true_slots
                pred_slot_count += pred_slots
                correct_slot_count += correct
            if label['role'] == 'assistant':
                if response.startswith('assistant:'):
                    pred = response[len('assistant:'):].strip()
                else:
                    pred = ''
                bleu_score = self._bleu4(pred, label['content'])
                bleu_scores.append(bleu_score)

        score_dict["slot_P"] = float(correct_slot_count/pred_slot_count) if pred_slot_count > 0 else 0
        score_dict["slot_R"] = float(correct_slot_count/true_slot_count) if true_slot_count > 0 else 0
        score_dict["slot_F1"] = 2*score_dict["slot_P"]*score_dict["slot_R"]/(score_dict["slot_P"]+score_dict["slot_R"]) if (score_dict["slot_P"]+score_dict["slot_R"]) > 0 else 0
        score_dict["bleu-4"] = float(np.mean(bleu_scores))
        for k, v in score_dict.items():
            score_dict[k] = round(v * 100, 4)
        print(f"score dict: {score_dict}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, required=True, help="main model weights")
    parser.add_argument("--ckpt", type=str, default=None, required=True, help="The checkpoint path")
    parser.add_argument("--data", type=str, default=None, required=True, help="The dataset file path")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, create_bnb_config())
    model = load_qlora(model, args.ckpt)
    evaluator = Evaluator(tokenizer, model, args.data)
    evaluator.compute_metrics()
