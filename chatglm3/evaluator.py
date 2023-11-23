import os
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def post_process(response):
    for response in response.split("<|assistant|>"):
        splited = response.split("\n", maxsplit=1)
        if len(splited) == 2:
            metadata, response = splited
        else:
            metadata = ""
            response = splited[0]
        if not metadata.strip():
            response = response.strip()
        else:
            response = "\n".join(response.split("\n")[1:-1])
            def tool_call(**kwargs):
                return kwargs
            if response:
                try:
                    parameters = eval(response)
                except:
                    parameters = {}
                response = {"name": metadata.strip(), "parameters": parameters}
    return response

def split_tokens(array):
    mapping = {64795:"<|user|>",64797:"<|observation|>"}
    delimiters = mapping.keys()
    turns = []
    idx = 0
    for i, num in enumerate(array):
        if num in delimiters:
            if i > idx:
                turns.append((array[idx:i],mapping[num]))
            idx = i + 1
    if idx < len(array):
        turns.append((array[idx:], "<|user|>"))
    return turns

def remove_minus100(ids,val):
    """
        -100是HF预留的id（不参与loss计算）
        有的tokenizer在decode -100时会报错
        因此在decode之前去除（替换为pad_id）
    """
    ids = np.array(ids)
    ids = np.where(ids == -100, val, ids)
    return ids

class Evaluator:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer

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

    def compute_metrics(self, eval_preds):
        score_dict = {
            "slot_P": 0,
            "slot_R": 0,
            "slot_F1": 0,
            "bleu-4": 0,
        }
        bleu_scores = []
        true_slot_count = 0
        pred_slot_count = 0
        correct_slot_count = 0

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # import pdb; pdb.set_trace()
        preds = np.argmax(preds, axis=-1)
        labels = remove_minus100(labels,self.tokenizer.pad_token_id)

        for i in range(preds.shape[0]):
            _preds, _labels = [], []
            label_turns = split_tokens(labels[i])
            for turn in label_turns:
                response = self.tokenizer.decode(turn[0], skip_special_tokens=True)
                response = post_process(response)
                _labels.append((response,turn[1]))
            tokens = preds[i]
            tokens = tokens[np.where(tokens==64796)[0][0]+1:]
            pred_turns = split_tokens(tokens)
            for turn in pred_turns[:len(label_turns)]:
                response = self.tokenizer.decode(turn[0], skip_special_tokens=True)
                response = post_process(response)
                _preds.append((response,turn[1]))
            for _pred_turn, _label_turn in zip(_preds, _labels):
                # next role is 'observation' mean current role is 'tool'
                if _label_turn[1] == '<|observation|>':
                    if _pred_turn[1] == '<|observation|>' and isinstance(_pred_turn[0], dict):
                        correct, pred_slots, true_slots = self._slot_accuracy(_pred_turn[0], _label_turn[0])
                    else:
                        correct, pred_slots, true_slots = self._slot_accuracy({}, _label_turn[0])
                    true_slot_count += true_slots
                    pred_slot_count += pred_slots
                    correct_slot_count += correct
                if _label_turn[1] == '<|user|>':
                    if _pred_turn[1] == '<|user|>' and isinstance(_pred_turn[0], str):
                        score = self._bleu4(_pred_turn[0], _label_turn[0])
                    else:
                        score = self._bleu4("", _label_turn[0])
                    bleu_scores.append(score)
        score_dict["slot_P"] = float(correct_slot_count/pred_slot_count) if pred_slot_count > 0 else 0
        score_dict["slot_R"] = float(correct_slot_count/true_slot_count) if true_slot_count > 0 else 0
        score_dict["slot_F1"] = 2*score_dict["slot_P"]*score_dict["slot_R"]/(score_dict["slot_P"]+score_dict["slot_R"]) if (score_dict["slot_P"]+score_dict["slot_R"]) > 0 else 0
        score_dict["bleu-4"] = sum(bleu_scores)/len(bleu_scores)
        for k, v in score_dict.items():
            score_dict[k] = round(v * 100, 4)
        return score_dict
        

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
