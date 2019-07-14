import vrep
import math
import time
import sys
import numpy as np 

from regression import generate_use_data
# from tensorflow import keras


class Hexapod(object):
    pass

    def __init__(self, client_id):
        self.client_id=client_id
        # handle init
        self.joint_handle = []
        prefix = ['left_joint', 'right_joint']
        for pre in range(2):
            for i in range(3):
                tmp = []
                for j in range(3):
                    object_name = '{0}{1}_{2}'.format(prefix[pre], j, i)
                    status,handle = vrep.simxGetObjectHandle(
                        client_id, object_name, vrep.simx_opmode_blocking)
                    if status!=0:
                        raise Exception("init error when get handle") 
                    tmp.append(handle)
                self.joint_handle.append(tmp)

        status,self.body=vrep.simxGetObjectHandle(self.client_id,'hexapod',vrep.simx_opmode_blocking)
        if status!=0:
            raise Exception("init error when get handle")

        self.left_0 = 0
        self.left_1 = 1
        self.left_2 = 2
        self.right_0 = 3
        self.right_1 = 4
        self.right_2 = 5

    def radian(self, degre):
        rad = degre*(math.pi/180)
        return rad

    def init(self):
        c=0
        a=25
        b=-70      
        for i in range(6):
            self.set_leg_position(i,[c,a,b])  

    def set_leg_position(self,addr,pos):
        try:
            for i in range(3):
                self.set_joint_position(addr,i,pos[i])
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

    def step_init(self):
        #load data set
        self.back_data=np.load('./data/NN_back_use.npy')
        self.middle_data=np.load('./data/NN_middle_use.npy')
        self.ahead_data=np.load('./data/NN_ahead_use.npy')
        # self.ahead_data=self.ahead_data[::-1]
        # self.back_data=self.back_data[::-1]

        #init robot
        self.init()

    def step(self):
        #back
        self.set_leg_position(1,np.concatenate([[40],self.middle_data[total_step][1:]]))
        self.set_leg_position(3,np.concatenate([[40],self.back_data[total_step][1:]]))
        self.set_leg_position(5,np.concatenate([[40],self.ahead_data[total_step][1:]]))
        for i in range(total_step):
            #ahead
            self.set_leg_position(0,self.back_data[i])
            self.set_leg_position(2,self.ahead_data[i])
            self.set_leg_position(4,self.middle_data[i])
            if i==0:
                time.sleep(0.5)
            time.sleep(0.01)

        #back
        self.set_leg_position(4,np.concatenate([self.middle_data[0][0],[60],self.middle_data[0][2]]))
        self.set_leg_position(0,np.concatenate([self.back_data[0][0],[60],self.back_data[0][1:]]))
        self.set_leg_position(2,np.concatenate([self.ahead_data[0][0],[60],self.ahead_data[0][1:]]))
        for i in range(total_step):
            #ahead
            self.set_leg_position(1,self.middle_data[i])
            self.set_leg_position(3,self.back_data[i])
            self.set_leg_position(5,self.ahead_data[i])
            if i==0:
                time.sleep(0.5)
            time.sleep(0.01)

    def test(self):
        hang_up=80

        total_step=self.ahead_data.shape[0]-1
        while True:
            for i in range(total_step):
                #ahead
                self.set_leg_position(0,self.ahead_data[i])
                self.set_leg_position(2,self.back_data[i])
                self.set_leg_position(4,self.middle_data[i])
                #back
                self.set_leg_position(1,np.concatenate([[self.middle_data[0][0]],[70],[self.middle_data[0][2]]]))
                self.set_leg_position(3,np.concatenate([[self.ahead_data[0][0]],[70],[self.ahead_data[0][2]]]))
                self.set_leg_position(5,np.concatenate([[self.back_data[0][0]],[70],[self.back_data[0][2]]]))
                if i==0:
                    time.sleep(0.05)
                time.sleep(0.003)
            for i in range(total_step):
                #ahead
                self.set_leg_position(1,self.middle_data[i])
                self.set_leg_position(3,self.ahead_data[i])
                self.set_leg_position(5,self.back_data[i])
                #back
                self.set_leg_position(0,np.concatenate([[self.ahead_data[0][0]],[70],[self.ahead_data[0][2]]]))
                self.set_leg_position(2,np.concatenate([[self.back_data[0][0]],[70],[self.back_data[0][2]]]))
                self.set_leg_position(4,np.concatenate([[self.middle_data[0][0]],[70],[self.middle_data[0][2]]]))
                if i==0:
                    time.sleep(0.05)
                time.sleep(0.003)

    def wait_for_stop(self,vrep_handle):
        speed_list=[]
        object_speed=10
        start_time=vrep.simxGetLastCmdTime(self.client_id)
        time.sleep(0.1)
        while object_speed > 0.01 :
            _,object_speed,_ = vrep.simxGetObjectVelocity(self.client_id,vrep_handle,vrep.simx_opmode_blocking)
            speed=np.array(object_speed)
            speed=np.sqrt(np.sum(np.power(object_speed,2)))
            object_speed=float(speed)
            speed_list.append(object_speed)
        end_time=vrep.simxGetLastCmdTime(self.client_id)
        speed_list=np.array(speed_list)
        # speed_sum=np.power(speed_list,2)
        # speed_sum=np.sqrt(np.sum(speed_sum,axis=1))
        speed_mean=float(np.mean(speed_list))
        used_time=end_time-start_time
        return speed_mean,used_time


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
    # generate_use_data([[-10,45],[-30,50],[-50,40]],50)
    client_id=connect(10)
    vrep.simxStartSimulation(client_id,vrep.simx_opmode_blocking)
    rb=Hexapod(client_id)
    rb.step_init()
    time.sleep(1)
    rb.test()

    

    
