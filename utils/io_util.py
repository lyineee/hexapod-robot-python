import json
import numpy as np


def read_json(path):
    data = []
    with open(path, 'r') as f:
        for tmp in f.readlines():
            data.append(json.loads(tmp))
    return data


def write_json(data, path='./result.json'):
    with open(path, 'w') as f:
        for i in data:
            json.dump(i, f)
            f.write('\n')


def gen_full_arr(data, hang_row):
    go_list = []
    back_list=[]
    for i, element in enumerate(data):
        go_list.append(element)
    for i, element in enumerate(data):
        hang = hang = (-0.0044444*i**2+0.133333*i)*hang_row
        back_list.append(element+hang)
    back_list.reverse()
    return np.array(go_list+back_list)


if __name__ == "__main__":
    read_json('./byopt_logs/logs_0.json')
