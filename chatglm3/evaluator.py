import os
import numpy as np
#from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import re

def remove_minus100(ids,val):
    """
        -100是HF预留的id（不参与loss计算）
        有的tokenizer在decode -100时会报错
        因此在decode之前去除（替换为pad_id）
    """
    ids = np.array(ids)
    ids = np.where(ids == -100, val, ids)
    return ids

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
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer

    def _bleu4(self,pred,label):
        pred = pred.strip()
        label = label.strip()
        
        hypothesis = list(pred)
        reference = list(label)

        if len(hypothesis) == 0 or len(reference) == 0:
            return 0

        bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
        return bleu_score

    def _slot_count(self,json_label):
        count = 0
        if json_label is not None:
            for _, v in json_label.items():
                if isinstance(v,list):
                    count += len(v)
                else:
                    count += 1
        return count

    def _slot_accuracy(self,pred,label):
        pred = pred.strip()
        label = label.strip()
        pred = parse_json(pred)
        label = parse_json(label)
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
        
        pred_slots = self._slot_count(pred)
        true_slots = self._slot_count(label)

        return correct, pred_slots, true_slots

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = remove_minus100(labels,self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

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
        
        for pred, label in zip(decoded_preds, decoded_labels):
            pred = pred.strip()
            label = label.strip()

              
            if pred.startswith("assistant:") and label.startswith("assistant:"):
                # 评估两个回复句子的BLEU SCORE  
                start = len("assistant:")
                bleu_score = self._bleu4(pred[start:],label[start:])
                bleu_scores.append(bleu_score)
            elif label.startswith("assistant:"):
                # 本次BLEU SCORE为0
                bleu_scores.append(0)
            else:
                # 尝试计算NLU的槽识别率
                correct, pred_slots, true_slots = self._slot_accuracy(pred,label)
                true_slot_count += true_slots
                pred_slot_count += pred_slots
                correct_slot_count += correct
        
        score_dict["slot_P"] = float(correct_slot_count/pred_slot_count) if pred_slot_count > 0 else 0
        score_dict["slot_R"] = float(correct_slot_count/true_slot_count) if true_slot_count > 0 else 0
        score_dict["slot_F1"] = 2*score_dict["slot_P"]*score_dict["slot_R"]/(score_dict["slot_P"]+score_dict["slot_R"]) if (score_dict["slot_P"]+score_dict["slot_R"]) > 0 else 0
        score_dict["bleu-4"] = float(np.mean(bleu_scores))
        for k, v in score_dict.items():
            score_dict[k] = round(v * 100, 4)
        return score_dict

def replace_all(input_str,src,tgt):
        while src != '' and src in input_str:
            input_str = input_str.replace(src,tgt)
        return input_str

def save_predictions(predict_results, tokenizer, output_dir):
    predictions = tokenizer.batch_decode(
            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    predictions = [pred.strip() for pred in predictions]
    label_ids = remove_minus100(predict_results.label_ids,tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(
        label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    labels = [label.strip() for label in labels]
    output_prediction_file = os.path.join(output_dir, "generated_predictions.txt")
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        for p, l in zip(predictions, labels):
            res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
            writer.write(f"{res}\n")
