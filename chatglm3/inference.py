import argparse
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default=None, required=True, help="The checkpoint path")
parser.add_argument("--model_path", type=str, default=None, required=True, help="main model weights")

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(args.model_path, config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(args.ckpt_path, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
model = model.to('cuda')

def chat(query, history, role):
    eos_token_id = [tokenizer.eos_token_id, 
                    tokenizer.get_command("<|user|>"), 
                    tokenizer.get_command("<|observation|>")]
    inputs = tokenizer.build_chat_input(query, history=history, role=role)
    inputs = inputs.to('cuda')
    outputs = model.generate(**inputs, max_length=4096, eos_token_id=eos_token_id)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
    response = tokenizer.decode(outputs)
    history.append({"role": role, "content": query})
    for response in response.split("<|assistant|>"):
        metadata, response = response.split("\n", maxsplit=1)
        if not metadata.strip():
            response = response.strip()
            history.append({"role": "assistant", "metadata": metadata, "content": response})
        else:
            history.append({"role": "assistant", "metadata": metadata, "content": response})
            response = "\n".join(response.split("\n")[1:-1])
            def tool_call(**kwargs):
                return kwargs
            parameters = eval(response)
            response = {"name": metadata.strip(), "parameters": parameters}
    return response, history

############ system message ##############
system_message = {'role': 'system', 'content': 'Answer the following questions as best as you can. You have access to the following tools:\n[\n    "search_hotels: 根据筛选条件查询酒店的函数\\nparameters: {\\"name\\":\\"酒店名称\\",\\"price_range_lower\\":\\"价格下限\\",\\"price_range_upper\\":\\"价格上限\\",\\"rating_range_lower\\":\\"评分下限\\",\\"rating_range_upper\\":\\"评分上限\\",\\"facilities\\": \\"酒店提供的设施\\"}\\noutput: 酒店信息dict组成的list"\n]'}
history = [system_message]

############ user message ##############
query = "请问可以帮我订一个500到800元的酒店吗?"
role = "user"
print(f"\nquery: {query}")

response, history = chat(query, history, role)
print(f"\ntool: {response}")

############ observation message ##############
results = [{"address": "北京海淀区大钟寺东路9号", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;国际长途电话;吹风机;24小时热水;西式餐厅;中式餐厅;残疾人设施;会议室;健身房;无烟房;酒吧;早餐服务;接站服务;接机服务;接待外宾;洗衣服务;行李寄存;叫醒服务", "hotel_id": 3, "name": "京仪大酒店", "phone": "010-62165588", "price": 685, "rating": 4.7, "subway": "知春路地铁站B口", "type": "高档型"}, {"address": "北京朝阳区霄云路26号", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;宽带上网;国际长途电话;吹风机;24小时热水;西式餐厅;中式餐厅;残疾人设施;室内游泳池;会议室;健身房;SPA;无烟房;商务中心;酒吧;早餐服务;接机服务;接待外宾;洗衣服务;行李寄存;叫醒服务", "hotel_id": 2, "name": "北京鹏润国际大酒店", "phone": "010-51086688", "price": 762, "rating": 4.6, "subway": "三元桥地铁站C2口", "type": "豪华型"}, {"address": "北京丰台区莲花池东路116-2号", "facilities": "酒店提供的设施:酒店各处提供wifi;宽带上网;国际长途电话;吹风机;24小时热水;暖气;会议室;无烟房;接待外宾;行李寄存;叫醒服务", "hotel_id": 1, "name": "如家快捷酒店", "phone": "010-63959988", "price": 558, "rating": 4.3, "subway": "北京西站地铁站A口", "type": "舒适型"}]
query = json.dumps(results, ensure_ascii=False)
role = "observation"

response, history = chat(query, history, role)
print(f"\nreply: {response.strip()}")
