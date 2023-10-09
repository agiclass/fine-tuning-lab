import json
import os
import random
import copy

def process_file(filename,data):
    with open(filename,'r',encoding="utf-8") as fp:
        dialog = json.load(fp)
        buffer = []
        for turn in dialog:
            if (turn["role"] == "assistant" or turn["role"] == "search") and len(buffer)>0:
                data.append({
                    "context" : copy.deepcopy(buffer),
                    "response" : turn
                })
            buffer.append(turn)
    return data

def process_dir(dir_path,data):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            process_file(file_path,data)
    return data

def split_data(data,ratio):
    random.shuffle(data)
    dev_size = int(len(data)*ratio)
    test_size = dev_size
    train_size = len(data)-dev_size-test_size
    train_data = data[:train_size]
    dev_data = data[train_size:train_size+dev_size]
    test_data = data[train_size+dev_size:]
    return train_data, dev_data, test_data

def write_jsonl(data,filename):
    with open(filename,"w",encoding="utf-8") as fp:
        for example in data:
            json_str = json.dumps(example,ensure_ascii=False)
            fp.write(json_str+"\n")

def main(raw_data_path,output_dir=".",ratio=0.1):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data = []
    data = process_dir(raw_data_path,data)
    train_data, dev_data, test_data = split_data(data,ratio)
    write_jsonl(train_data,os.path.join(output_dir,"train.jsonl"))
    write_jsonl(dev_data,os.path.join(output_dir,"dev.jsonl"))
    write_jsonl(test_data,os.path.join(output_dir,"test.jsonl"))

main("enhanced_hotel_data")