import json

def read_json(path):
    data=[]
    with open(path,'r') as f:
        for tmp in f.readlines():
            data.append(json.loads(tmp))
    return data

def write_json(data,path='./result.json'):
    with open(path,'w') as f:
        for i in data:
            json.dump(i,f)
            f.write('\n') 

if __name__ == "__main__":
    read_json('./byopt_logs/logs_0.json')
    