import vrep
import time
import sys 
import pickle
import numpy as np 
import math


# from math import sin 
# from gaft import GAEngine
# from gaft.components import BinaryIndividual, Population
# from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation
# from gaft.analysis import ConsoleOutput,fitness_store

class Leg(object):
    pass

    def __init__(self, client_id):
        self.client_id=client_id
        # handle init
        self.joint_handle = []
        i=0
        tmp = []
        for j in range(3):
            object_name = '{0}{1}_{2}'.format('left_joint', j, i)
            status,handle = vrep.simxGetObjectHandle(
                client_id, object_name, vrep.simx_opmode_blocking)
            if status!=0:
                raise Exception("init error when get handle") 
            tmp.append(handle)
        self.joint_handle.append(tmp)

        status,self.tip_handle=vrep.simxGetObjectHandle(self.client_id,'tip',vrep.simx_opmode_blocking)
        if status!=0:
            raise Exception("init error when get handle")

        self.left_0 = 0
        self.left_1 = 1
        self.left_2 = 2
        self.right_0 = 3
        self.right_1 = 4
        self.right_2 = 5
        vrep.simxStartSimulation(self.client_id,vrep.simx_opmode_blocking)


    def radian(self, degre):
        rad = degre*(math.pi/180)
        return rad

    def degre(self,radian):
        deg=radian/(math.pi/180)
        return deg

    def init(self):
        c=0
        a=25
        b=-70      
        for i in range(1):
            print(i)
            self.set_leg_position(i,[c,a,b])  

    def set_leg_position(self,addr,pos):
        try:
            for i in range(3):
                self.set_joint_position(addr,i,pos[i])
        except:
            raise

    def set_leg_2_position(self,addr,pos):
        try:
            for i in range(1,3):
                self.set_joint_position(addr,i,pos[i-1])
        except:
            raise

    def set_joint_position(self,addr,joint,pos):
        #TODO exception handling
        if addr not in range(6):
            pass
        else:
            pos=self.radian(pos)
            status=vrep.simxSetJointTargetPosition(self.client_id,self.joint_handle[addr][joint],pos,vrep.simx_opmode_oneshot)
            if status !=0:
                pass

    def set_start_pos(self):
        self.set_leg_position(0,[-40,25,-70])
    
    def get_tip_pos(self):
        _,pos=vrep.simxGetObjectPosition(self.client_id,self.tip_handle,-1,vrep.simx_opmode_blocking)
        return pos[1]

    def get_joint_0_pos(self):
        _,pos=vrep.simxGetJointPosition(self.client_id,self.joint_handle[0][0],vrep.simx_opmode_blocking)
        return pos

    def init_sample(self):
        self.last_tip_pos=100
        self.label=[]
        self.data=[]

    def get_data(self):
        data=[]
        for i in range(3):
            status,pos=vrep.simxGetJointPosition(self.client_id,self.joint_handle[0][i],vrep.simx_opmode_blocking)
            if status!=0:
                raise Exception("init error when get handle") 
            else:
                data.append(self.degre(pos))
        return data[0],data[1:]

    def sample(self):
        init_pos=self.get_tip_pos()
        if abs(init_pos-self.last_tip_pos) > 1:
            label,data=self.get_data()
            self.label.append(label)
            self.data.append(data)
            # print(label)
        
    def sample_loop(self):
        while True:
            self.sample()

    def save_sample_data(self,name):
        print('there')
        label=np.array(self.label)
        data=np.array(self.data)

        np.save('./data/label_{}.npy'.format(name),label)
        np.save('./data/data_{}.npy'.format(name),data)



    def test(self):
        self.init()

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
    # vrep.simxStartSimulation(client_id,vrep.simx_opmode_blocking)
    rb=Leg(client_id)
    rb.init_sample()
    try:
        rb.sample_loop()
    except KeyboardInterrupt as e:
        rb.save_sample_data('back')