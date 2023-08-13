import os
import numpy as np
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json

class Evaluator:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": [],
            "whole_sentence_acc": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            pred = pred.strip()
            label = label.strip()
            if pred == label :
                score_dict["whole_sentence_acc"].append(1)
            else:
                score_dict["whole_sentence_acc"].append(0)

            hypothesis = list(pred)
            reference = list(label)

            if len(hypothesis) == 0 or len(reference) == 0:
                for k, v in score_dict.items():
                    if k.startswith('rouge') or k.startswith('bleu'):
                        score_dict[k].append(0.0)
                continue

            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
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
    labels = tokenizer.batch_decode(
        predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    labels = [label.strip() for label in labels]
    output_prediction_file = os.path.join(output_dir, "generated_predictions.txt")
    with open(output_prediction_file, "w", encoding="utf-8") as writer:
        for p, l in zip(predictions, labels):
            res = json.dumps({"labels": replace_all(l,'<image_-100>',''), "predict": p}, ensure_ascii=False)
            writer.write(f"{res}\n")