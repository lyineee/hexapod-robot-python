from hexapod import Hexapod
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

if __name__ == "__main__":
    client_id=connect(10)
    vrep.simxStartSimulation(client_id,vrep.simx_opmode_blocking)
    rb=Hexapod(client_id)
    rb.test()