import json

def build_prompt(context):
    prompt = ""
    for i, turn in enumerate(context):
        if turn["role"] in ["user","return"]:
            prompt += f"[Round {i}]\n\n"
        if turn["role"] in ["user","assistant"]:
            prompt += turn["role"] + ": " + turn["content"] + "\n\n"
        else:
            if turn["role"] == "search":
                obj = turn["arguments"]
                filtered_obj = {k: v for k, v in obj.items() if v is not None}
                prompt += turn["role"] + ":\n" + json.dumps(filtered_obj,indent=4,ensure_ascii=False) + "\n\n"
            else:
                obj = turn["records"]
                prompt += turn["role"] + ":\n" + json.dumps(obj,indent=4,ensure_ascii=False) + "\n\n"   
            
    return prompt

def build_response(response):
    if response["role"] == "assistant":
        return "assistant: " + response["content"]
    else:
        obj = response["arguments"]
        filtered_obj = {k: v for k, v in obj.items() if v is not None}
        return "search:\n" + json.dumps(filtered_obj,indent=4,ensure_ascii=False)