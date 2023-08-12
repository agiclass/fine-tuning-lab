import json
import sys

def convert_to_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []
    for item in data["data"]:
        paragraphs = item["paragraphs"]
        for qa in paragraphs[0]["qas"]:
            question = qa["question"]
            if qa["is_impossible"] == "true":
                output.append({"input": "判例:\n" + paragraphs[0]["context"] + "\n问题:\n" + question + "\n答案:\n", "output": "无答案"})
                continue

            for answer in qa["answers"]:
                answer_text = answer["text"]
                output.append({"input": "判例:\n" + paragraphs[0]["context"] + "\n问题:\n" + question + "\n答案:\n", "output": answer_text})
                break # 只要第一个答案

    with open(output_file, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_to_jsonl(input_file, output_file)

