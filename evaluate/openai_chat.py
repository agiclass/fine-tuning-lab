import os
import openai
import config
import time
openai.organization = config.OPENAI_ORG
openai.api_key = config.OPENAI_API_KEY

def chat():
    completion = openai.ChatCompletion.create(
      model=config.MODEL,
      messages=[
        {"role": "system", "content": "你是一个熟悉中国法律的专家"},
        {"role": "user", "content": "经审理查明,原告孙x0与被告吴2婚后于2009年1月3日生育长子孙某乙2012年5月24日,原被告自愿离婚,协议约定婚生长子孙某乙跟随女方(被告吴2)生活,抚养费自理2012年10月12日,双方又生育次子孙x6,但在原告起诉之前,两个孩子实际上一直跟随原告生活,由原告抚养现原告向本院提起诉讼,请求依法判令双方婚生长子孙某乙、婚生次子孙x6由原告抚养,被告每月支付1000元抚养费,至两个孩子年满十八周岁止 问题:离婚后长子的抚养权归谁所有？"}
      ]
    )

    print(completion.choices[0].message)

if __name__ == "__main__":
    chat()

