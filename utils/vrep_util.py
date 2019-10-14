import vrep
import time
import sys


def connect(retry):
    while True:
        # vrep.simxFinish(-1)  # 关掉之前的连接
        clientId = vrep.simxStart(
            "127.0.0.1", 19997, True, True, 100, 5)  # 建立和服务器的连接
        if clientId != -1:  # 连接成功
            print('connect successfully')
            return clientId
        elif retry > 0:
            retry -= 1
        else:
            print('connect time out')
            sys.exit(1)
        time.sleep(1)


def map(input_num,input_range,output_range):
    a=(output_range[1]-output_range[0])/(input_range[1]-input_range[0])
    b=output_range[0]-input_range[0]*a
    result=a*input_num+b
    return result