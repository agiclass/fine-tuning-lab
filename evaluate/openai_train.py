import os
import openai
import config
import time
openai.organization = config.OPENAI_ORG
openai.api_key = config.OPENAI_API_KEY

def train():
    # upload files 
    train_file = openai.File.create( file=open("../data/openai/train.jsonl", "rb"), purpose='fine-tune')
    print(train_file)
    dev_file = openai.File.create( file=open("../data/openai/dev.jsonl", "rb"), purpose='fine-tune')
    print(dev_file)

    # waiting files ready 
    while True:
        time.sleep(1)
        status = openai.File.retrieve(train_file["id"])
        if status["status"] == "processed":
            break


    while True:
        time.sleep(1)
        status = openai.File.retrieve(dev_file["id"])
        if status["status"] == "processed":
            break

    # create jobs for fine-tune 
    try:
        ft_obj = openai.FineTuningJob.create(training_file=train_file["id"], validation_file=dev_file["id"], model="gpt-3.5-turbo")
        print(ft_obj)
    except e:
        print(e)
        del_file = openai.File.delete(train_file["id"])
        print(del_file)
        del_file = openai.File.delete(dev_file["id"])
        print(del_file)



if __name__ == "__main__":
    train()

