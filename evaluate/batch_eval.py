from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
import openai
import sys
import numpy as np
import requests
import json
import config

# Metric
def compute_metrics(decoded_preds, decoded_labels):

    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": [],
        "whole_sentence_acc": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        print("pred:", pred, "label", label)
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

def chatglm_check_result(query, result):

    url = config.CHATGLM_URL

    payload = json.dumps({
        "prompt": query,
        "history": [],
        "max_length": 2048,
        "top_p": 0.7,
        "temperature": 0.05
        })
    headers = {
            'Content-Type': 'application/json'
            }

    response = requests.request("POST", url, headers=headers, data=payload)
    #print(result, json.loads(response.text)["response"])
    return compute_metrics([result], [json.loads(response.text)["response"]])

def openai_check_result(query, result):

    openai.organization = config.OPENAI_ORG
    openai.api_key = config.OPENAI_API_KEY
    query = query.replace("\n","")
    completion = openai.ChatCompletion.create(
      model=config.MODEL,
      messages=[
        {"role": "system", "content": "你是一个熟悉中国法律的专家"},
        {"role": "user", "content": query }
      ]
    )

    #print(result, completion.choices[0].message)

    return compute_metrics([result], [completion.choices[0].message["content"]])
    #return None


def file_to_check(filename, method):
    scores = {'rouge-1': 0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu-4': 0.0, 'whole_sentence_acc': 0.0}
    with open(filename, "r", encoding="utf-8") as f:
        times = 0
        for data in f.readlines():
            line = json.loads(data)
            times += 1

            prompt = line["input"]
            label = line["output"]

            score_item = scores
            if method == "chatglm":
                score_item = chatglm_check_result(prompt, label)
            elif method == "openai":
                score_item = openai_check_result(prompt, label)
            else:
                print("method not support")

            print("score:", score_item)
            for k, v in score_item.items():
                scores[k] += v
            print(scores)
            
        print("times:", times)
        for k, v in scores.items():
            print(k, v)
            scores[k] = v/times
        print(scores)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval_test.py filename method[chatglm|openai]")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2]
    file_to_check(filename, method)
