import vrep
import math
import time
import sys


class Hexapod(object):
    pass

    def __init__(self, client_id):
        self.client_id=client_id
        # handle init
        self.left_joint = []
        self.right_joint = []
        prefix = ['left_joint', 'right_joint']
        for pre in prefix:
            for i in range(3):
                tmp = []
                for j in range(3):
                    object_name = '{0}{1}_{2}'.format(pre, i, j)
                    status,handle = vrep.simxGetObjectHandle(
                        client_id, object_name, vrep.simx_opmode_blocking)
                    if status!=0:
                        raise Exception("init error when get handle") 
                    tmp.append(handle)
                if pre == 'left_joint':
                    self.left_joint.append(tmp)
                else:
                    self.right_joint.append(tmp)

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
        

        self.set_left_joint(0,0,c)
        self.set_left_joint(0,1,c)
        self.set_left_joint(0,2,c)
        self.set_right_joint(0,0,c)
        self.set_right_joint(0,1,c)
        self.set_right_joint(0,2,c)

        self.set_left_joint(1,0,a)
        self.set_left_joint(1,1,a)
        self.set_left_joint(1,2,a)
        self.set_right_joint(1,0,a)
        self.set_right_joint(1,1,a)
        self.set_right_joint(1,2,a)

        self.set_left_joint(2,0,b)
        self.set_left_joint(2,1,b)
        self.set_left_joint(2,2,b)
        self.set_right_joint(2,0,b)
        self.set_right_joint(2,1,b)
        self.set_right_joint(2,2,b)


    def set_left_joint(self,joint_a,joint_b,position):
        pos=self.radian(position)
        vrep.simxSetJointTargetPosition(self.client_id,self.left_joint[joint_a][joint_b],pos,vrep.simx_opmode_oneshot)

    def set_right_joint(self,joint_a,joint_b,position):
        pos=self.radian(position)
        vrep.simxSetJointTargetPosition(self.client_id,self.right_joint[joint_a][joint_b],pos,vrep.simx_opmode_oneshot)

    def set_leg_position(self,addr,pos0,po1,po2):
        if addr not in range(6):
            pass
        

    def test(self):
        a=30
        self.set_left_joint(0,0,a)
        self.set_right_joint(0,1,a)
        self.set_left_joint(0,2,a)

        time.sleep(0.3)
        # self.set_right_joint(0,0,45)
        # self.set_left_joint(0,1,45)
        # self.set_right_joint(0,2,45)

    def step(self,inverse=1):
        pass



    def half_step(self,addr,addr1,position):
        if addr==1:
            self.set_right_joint(addr1,0,position)
            self.set_left_joint(addr1,1,position)
            self.set_right_joint(addr1,2,position)
        else:
            self.set_left_joint(addr1,0,position)
            self.set_right_joint(addr1,1,position)
            self.set_left_joint(addr1,2,position)

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
    rb.init()
    time.sleep(1)
    while True:
        rb.step()
    # vrep.simxSetJointTargetPosition(rb.client_id,rb.left_joint[1][0],90,vrep.simx_opmode_blocking)

    

    
