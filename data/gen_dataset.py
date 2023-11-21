import os
import json
import random

def split_data(data, ratio=0.1):
    random.shuffle(data)
    dev_size = int(len(data)*ratio)
    test_size = dev_size
    train_size = len(data)-dev_size-test_size
    train_data = data[:train_size]
    dev_data = data[train_size:train_size+dev_size]
    test_data = data[train_size+dev_size:]
    return train_data, dev_data, test_data

def write_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            json_str = json.dumps(item,ensure_ascii=False)
            f.write(json_str + "\n")

tool_description = """
search_hotels: 根据筛选条件查询酒店的函数
parameters: {"name":"酒店名称","price_range_lower":"价格下限","price_range_upper":"价格上限","rating_range_lower":"评分下限","rating_range_upper":"评分上限","facilities": "酒店提供的设施"}
output: 酒店信息dict组成的list
"""

dataset = []
files = os.listdir("enhanced_hotel_data")
for file in files:
    data = {"tools":[tool_description.strip()],"conversations":[]}
    with open(f"enhanced_hotel_data/{file}","r") as f:
        dialog = json.load(f)
    for turn in dialog:
        if turn["role"] == "search":
            think = {"role":"assistant","content":"我需要使用search_hotels工具来查询酒店"}
            data["conversations"].append(think)
            action = {"role":"tool","name":"search_hotels","parameters":turn["arguments"]}
            data["conversations"].append(action)
        elif turn["role"] == "return":
            data["conversations"][-1]["observation"] = turn["records"]
        else:
            data["conversations"].append(turn)
    dataset.append(data)

train_data, dev_data, test_data = split_data(dataset)
write_jsonl(train_data, os.path.join("train.jsonl"))
write_jsonl(dev_data, os.path.join("dev.jsonl"))
write_jsonl(test_data, os.path.join("test.jsonl"))
