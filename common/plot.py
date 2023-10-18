# coding=utf-8

import matplotlib.pyplot as plt
import ast
import sys
import numpy as np

def parse_equation(line):
    parts = line.split('=')
    if len(parts)!=2:
        return None, None
    try:
        val = float(parts[1].strip())
    except:
        val = None
    return parts[0].strip(), val

def extract_data(filename):
    data = {}
    with open(filename,'r',encoding='utf-8') as fp:
        final = False
        last_epoch = None
        for line in fp:
            line = line.strip()
            if line == "***** eval metrics *****":
                final = True
                continue
            if line == "***** predict metrics *****":
                break
            if final:
                left, right = parse_equation(line)
                if left is None or right is None:
                    continue
                if left == "epoch":
                    last_epoch = right
                elif left not in ["eval_runtime","eval_samples","eval_samples_per_second","eval_steps_per_second"]:
                    if left not in data:
                        data[left] = {}
                    data[left][last_epoch] = right
                continue

            point = None
            if "'epoch':" in line:
                start = line.find('{')
                end = line.find('}')
                if start == -1 or end == -1:
                    continue
                point = line[start:end+1]
            if point:
                point = ast.literal_eval(point)
                epoch = point['epoch']
                for k,v in point.items():
                    if k in ['epoch','learning_rate','train_loss','loss']:
                        continue
                    if k.endswith('runtime'):
                        continue
                    if k.endswith('samples_per_second'):
                        continue
                    if k.endswith('steps_per_second'):
                        continue
                    if k not in data:
                        data[k] = {}
                    data[k][epoch] = v
        
    return data

def get_xy(data):
    x = []
    y = []
    for k, v in data.items():
        x.append(float(k))
        y.append(float(v))
    return x,y


if __name__ == '__main__':
    log_file = sys.argv[1]
    data = extract_data(log_file)
    fig, ax_ = plt.subplots()
    
    ax = [ax_,ax_.twinx()]
    lines = []
    for k,v in data.items():
        x,y = get_xy(v)
        if 'bleu' in k:
            index = 1
            mark = '-.'
        else:
            index = 0
            mark = '-'
        line,  = ax[index].plot(x, y, mark, label = k.replace("eval_",""))
        lines.append(line)

    for i in range(2):
        if i == 0:
            ax[i].set_yticks(np.arange(50, 100, step=5))
            ax[i].set_ylabel('P / R / F1')
        else:
            ax[i].set_yticks(np.arange(30, 80, step=5))
            ax[i].set_ylabel('BLEU')

    ax[0].set_xlabel('EPOCHS')
    labels = [l.get_label() for l in lines]
    ax[0].legend(lines, labels, loc='lower right',fontsize='x-small')
    plt.savefig(log_file.replace(".txt","")+".png")