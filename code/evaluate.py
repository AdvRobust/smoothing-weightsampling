# coding=utf-8
import numpy as np


def obtain_ACR(path):
    predictions = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            tmp = line.replace('\n', '').split('\t')
            predictions.append((int(tmp[4]), float(tmp[3])))
    list_radius = []
    for i, tmp in enumerate(predictions):
        is_correct, radius = tmp[0], tmp[1]
        if is_correct == 1:
            list_radius.append(radius)
        else:
            list_radius.append(0.0)
    print(np.mean(list_radius))


def evaluation_radius(path):
    predictions = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            tmp = line.replace('\n', '').split('\t')
            predictions.append((int(tmp[4]), float(tmp[3])))
    count0 = 0
    count025 = 0
    count05 = 0
    count075 = 0
    count10 = 0
    count125 = 0
    count15 = 0
    count175 = 0
    count20 = 0
    count225 = 0
    for i, tmp in enumerate(predictions):
        is_correct, radius = tmp[0], tmp[1]
        if is_correct == 1:
            count0 += 1
            if radius >= 0.25:
                count025 += 1
            if radius >= 0.5:
                count05 += 1
            if radius >= 0.75:
                count075 += 1
            if radius >= 1.0:
                count10 += 1
            if radius >= 1.25:
                count125 += 1
            if radius >= 1.5:
                count15 += 1
            if radius >= 1.75:
                count175 += 1
            if radius >= 2.0:
                count20 += 1
            if radius >= 2.25:
                count225 += 1
    print("L=0.25: {}".format(count025 / len(predictions)))
    print("L=0.5: {}".format(count05 / len(predictions)))
    print("L=0.75: {}".format(count075 / len(predictions)))
    print("L=1.0: {}".format(count10 / len(predictions)))
    print("L=1.25: {}".format(count125 / len(predictions)))
    print("L=1.5: {}".format(count15 / len(predictions)))
    print("L=1.75: {}".format(count175 / len(predictions)))
    print("L=2.0: {}".format(count20 / len(predictions)))
    print("L=2.25: {}".format(count225 / len(predictions)))