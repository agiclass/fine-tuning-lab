# coding=utf-8

import matplotlib.pyplot as plt
import ast
import sys
import numpy as np

def extract_data(filename):
    data = {}
    with open(filename,'r',encoding='utf-8') as fp:
        for line in fp:
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
                    if k in ['epoch','learning_rate','train_loss']:
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
    fig, ax = plt.subplots(2)
    for k,v in data.items():
        x,y = get_xy(v)
        if max(y) <= 1:
            index = 0
        else:
            index = 1
        ax[index].plot(x, y, label = k)
    for i in range(2):
        if i == 0:
            ax[i].set_yticks(np.arange(0, 1, step=0.1))
        else:
            ax[i].set_xlabel('epochs')
        ax[i].legend(loc='lower right',fontsize='x-small')
    plt.savefig(log_file.replace(".txt","")+".png")

