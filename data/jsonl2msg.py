import json
import sys

def convert_jsonl(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        # 遍历输入文件的每一行，并转换格式
        for line in infile:
            data = json.loads(line.strip())

            # 生成三个role消息
            item = {}
            
            system_message = {
                "role": "system",
                "content": "你是一个熟悉中国法律的专家"
            }
            user_message = {
                "role": "user",
                "content": data["input"]
            }
            assistant_message = {
                "role": "assistant",
                "content": data["output"]
            }

            item["messages"] = [system_message, user_message, assistant_message]

            # 将三个消息写入输出文件
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script_name.py input_file.jsonl output_file.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    convert_jsonl(input_file, output_file)

