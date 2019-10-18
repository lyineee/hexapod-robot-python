import math
import threading
import time
from math import sqrt

import numpy as np

import vrep
from utils.vrep_util import connect, map

# from tensorflow import keras


class Hexapod(object):
    #lock define 
    lock=threading.Lock()

    #step data
    back_data = None
    middle_data = None
    ahead_data = None

    back_turn_data_1=None
    middle_turn_data_1=None
    ahead_turn_data_1=None

    back_turn_data_2=None
    middle_turn_data_2=None
    ahead_turn_data_2=None

    delay=None

    #state
    state=1

    def __init__(self, client_id=None):
        if client_id is None:
            try:
                self.client_id=connect(3)
            except SystemExit:
                print('connet fail, closing')
        else:
            self.client_id = client_id
        # handle init
        self.joint_handle = []
        prefix = ['left_joint', 'right_joint']
        for pre in range(2):
            for i in range(3):
                tmp = []
                for j in range(3):
                    object_name = '{0}{1}_{2}'.format(prefix[pre], j, i)
                    status, handle = vrep.simxGetObjectHandle(
                        client_id, object_name, vrep.simx_opmode_blocking)
                    if status != 0:
                        raise Exception("init error when get handle")
                    tmp.append(handle)
                self.joint_handle.append(tmp)

        status, self.body = vrep.simxGetObjectHandle(
            self.client_id, 'hexapod', vrep.simx_opmode_blocking)
        if status != 0:
            raise Exception("init error when get handle")

        status, self.vision_sensor = vrep.simxGetObjectHandle(
            self.client_id, 'vision_sensor', vrep.simx_opmode_blocking)
        if status != 0:
            raise Exception("init error when get handle")

        #image sensor init
        vrep.simxGetVisionSensorImage(self.client_id,self.vision_sensor,0,vrep.simx_opmode_streaming)

    def radian(self, degre):
        rad = degre*(math.pi/180)
        return rad

    def init(self):
        c = 0
        a = 25
        b = -70
        for i in range(6):
            self.set_leg_position(i, [c, a, b])

    def set_leg_position(self, addr, pos):
        self.lock.acquire()
        try:
            for i in range(3):
                self.set_joint_position(addr, i, pos[i])
        except:
            raise
        finally:
            self.lock.release()

    def set_joint_position(self, addr, joint, pos):
        # TODO exception handling
        if addr not in range(6):
            pass
        else:
            pos = self.radian(pos)
            status = vrep.simxSetJointTargetPosition(
                self.client_id, self.joint_handle[addr][joint], pos, vrep.simx_opmode_oneshot)
            if status != 0:
                pass

    def step_init(self):
        # load data set
        self.back_data = np.load('./data/NN_back_use.npy')
        self.middle_data = np.load('./data/NN_middle_use.npy')
        self.ahead_data = np.load('./data/NN_ahead_use.npy')
        
        self.back_turn_data_1 = np.load('./data/NN_back_turn_use_1.npy')
        self.middle_turn_data_1 = np.load('./data/NN_middle_turn_use_1.npy')
        self.ahead_turn_data_1 = np.load('./data/NN_ahead_turn_use_1.npy')

        self.back_turn_data_2 = np.load('./data/NN_back_turn_use_2.npy')
        self.middle_turn_data_2 = np.load('./data/NN_middle_turn_use_2.npy')
        self.ahead_turn_data_2 = np.load('./data/NN_ahead_turn_use_2.npy')
    

    def set_step_data(self, data,data_turn,delay=-1):
        self.back_data = np.array(data[0])
        self.middle_data = np.array(data[1])
        self.ahead_data = np.array(data[2])

        self.back_turn_data_1 = np.array(data_turn[0])
        self.middle_turn_data_1 = np.array(data_turn[1])
        self.ahead_turn_data_1 = np.array(data_turn[2])

        if delay!=-1:
            self.delay=delay

    def keep_step(self,time_0,q,direction=None):
        #wait for start
        q.get()
        while True:
            try:
                if direction is None:
                    self.one_step_t(time_0)
                else:
                    direction()
                if not q.empty():
                    break
            except:
                break
    
    def step(self,data):
        assert(self.delay is not None)
        delay=self.delay

        stage=self.state

        total_step = data[0].shape[0]
        hang = 30
        full_flag=False
        if stage==-1:
            full_flag=True
        index=0
        l=total_step-1
        
        if stage==1 or full_flag:
            for i in range(total_step):
                if i in [0]:
                    continue
                hang=(-(4/l**2)*i**2+(4/l)*i)*10
                # ahead
                self.set_leg_position(0, data[0][i])
                self.set_leg_position(2, data[2][i])
                self.set_leg_position(4, data[4][i])

                # back hang up
                self.set_leg_position(1, data[1][total_step-i-1]+hang)
                self.set_leg_position(3, data[3][total_step-i-1]+hang)
                self.set_leg_position(5, data[5][total_step-i-1]+hang)
                time.sleep(delay)
                index+=1
        index=0
        if stage==2 or full_flag:
            for i in range(total_step):
                if i in [0]:
                    continue
                hang=(-(4/l**2)*i**2+(4/l)*i)*10
                # ahead
                self.set_leg_position(1, data[1][i])
                self.set_leg_position(3, data[3][i])
                self.set_leg_position(5, data[5][i])
    
                # back hang up
                self.set_leg_position(0, data[0][total_step-i-1]+hang)
                self.set_leg_position(2, data[2][total_step-i-1]+hang)
                self.set_leg_position(4, data[4][total_step-i-1]+hang)
                time.sleep(delay)
                index+=1

    def right_1(self,delay=-1,stage=-1):
        # self.delay=delay
        self.step([self.ahead_data,self.middle_data,self.back_data,self.ahead_turn_data_1,self.middle_turn_data_1,self.back_turn_data_1])
        self.change_state()

    def right_2(self,delay=-1,stage=-1):
        # self.delay=delay
        self.step([self.ahead_data,self.middle_data,self.back_data,self.ahead_turn_data_2,self.middle_turn_data_2,self.back_turn_data_2])
        self.change_state()

    def left_1(self,delay=-1,stage=-1):
        # self.delay=delay
        self.step([self.ahead_turn_data_1,self.middle_turn_data_1,self.back_turn_data_1,self.ahead_data,self.middle_data,self.back_data])
        self.change_state()

    def left_2(self,delay=-1,stage=-1):
        # self.delay=delay
        self.step([self.ahead_turn_data_2,self.middle_turn_data_2,self.back_turn_data_2,self.ahead_data,self.middle_data,self.back_data])
        self.change_state()

    def ahead(self,delay=-1,stage=-1):
        # self.delay=delay
        self.step([self.ahead_data,self.middle_data,self.back_data,self.ahead_data,self.middle_data,self.back_data])
        self.change_state()

    def one_step_t(self, time_0,stage=-1):
        total_step = self.ahead_data.shape[0]
        hang = 20
        full_flag=False
        if stage==-1:
            full_flag=True
        index=0
        l=total_step-1
        
        if stage==1 or full_flag:
            for i in range(total_step):
                # if i in [0,1,2] :
                #     continue
                if i in [0]:
                    continue
                hang=(-(4/l**2)*i**2+(4/l)*i)*10
                # ahead
                self.set_leg_position(0, self.ahead_data[i])
                self.set_leg_position(2, self.back_data[i])
                self.set_leg_position(4, self.middle_data[i])

                # back hang up
                # hang=-0.04444*(index**2)+1.3333*index
                self.set_leg_position(1, self.middle_data[total_step-i-1]+hang)
                self.set_leg_position(3, self.ahead_data[total_step-i-1]+hang)
                self.set_leg_position(5, self.back_data[total_step-i-1]+hang)
                time.sleep(time_0)
                index+=1
        index=0
        if stage==2 or full_flag:
            for i in range(total_step):
                # if i in [0,1,2]:
                #     continue
                if i in [0]:
                    continue
                hang=(-(4/l**2)*i**2+(4/l)*i)*10
                # ahead
                self.set_leg_position(1, self.middle_data[i])
                self.set_leg_position(3, self.ahead_data[i])
                self.set_leg_position(5, self.back_data[i])
    
                # back hang up
                # hang=-0.04444*(index**2)+1.3333*index
                self.set_leg_position(0, self.ahead_data[total_step-i-1]+hang)
                self.set_leg_position(2, self.back_data[total_step-i-1]+hang)
                self.set_leg_position(4, self.middle_data[total_step-i-1]+hang)
                time.sleep(time_0)
                index+=1


    def turn_right(self, right_range,stage=-1,time_0 = 0.005):
        total_step = self.ahead_data.shape[0]-1
        turn_step = max(right_range)-min(right_range)
        hang = 10
        hang_hight = 10
        full_flag=False
        if stage==-1:
            full_flag=True
        l=total_step-1

        if stage==1 or full_flag:
            for i in range(total_step):
                hang=(-(4/l**2)*i**2+(4/l)*i)*hang_hight
                # ahead
                self.set_leg_position(0, self.ahead_data[i])
                self.set_leg_position(1, self.middle_data[total_step-i]+hang)
                self.set_leg_position(2, self.back_data[i])

                # back
                if i < turn_step:
                    self.set_leg_position(
                        3, self.ahead_data[right_range[0]:right_range[1]][turn_step-1-i]+hang)
                    self.set_leg_position(
                        4, self.middle_data[right_range[0]:right_range[1]][i])
                    self.set_leg_position(
                        5, self.back_data[right_range[0]:right_range[1]][turn_step-1-i]+hang)
                time.sleep(time_0)

        if stage==2 or full_flag:
            for i in range(total_step):
                hang=(-(4/l**2)*i**2+(4/l)*i)*hang_hight
                # ahead
                if i < turn_step:
                    self.set_leg_position(
                        3, self.ahead_data[right_range[0]:right_range[1]][i])
                    self.set_leg_position(
                        4, self.middle_data[right_range[0]:right_range[1]][turn_step-1-i]+hang)
                    self.set_leg_position(
                        5, self.back_data[right_range[0]:right_range[1]][i])
    
                # back
                self.set_leg_position(0, self.ahead_data[total_step-i]+hang)
                self.set_leg_position(1, self.middle_data[i])
                self.set_leg_position(2, self.back_data[total_step-i]+hang)
                time.sleep(time_0)

    def turn_left(self, left_range,stage=-1,time_0 = 0.005):
        total_step = self.ahead_data.shape[0]-1
        turn_step = max(left_range)-min(left_range)
        hang = 10
        hang_hight=10
        full_flag=False
        if stage==-1:
            full_flag=True
        l=total_step-1

        if stage == 1 or full_flag:
            for i in range(total_step):
                hang=(-(4/l**2)*i**2+(4/l)*i)*hang_hight
                if i < turn_step:
                    self.set_leg_position(
                        0, self.ahead_data[left_range[0]:left_range[1]][i])
                    self.set_leg_position(
                        1, self.middle_data[left_range[0]:left_range[1]][turn_step-1-i]+hang)
                    self.set_leg_position(
                        2, self.back_data[left_range[0]:left_range[1]][i])

                # back
                self.set_leg_position(3, self.ahead_data[total_step-i]+hang)
                self.set_leg_position(4, self.middle_data[i])
                self.set_leg_position(5, self.back_data[total_step-i]+hang)
                time.sleep(time_0)

        if stage == 2 or full_flag:
            for i in range(total_step):
                hang=(-(4/l**2)*i**2+(4/l)*i)*hang_hight
                # ahead
                self.set_leg_position(3, self.ahead_data[i])
                self.set_leg_position(4, self.middle_data[total_step-i]+hang)
                self.set_leg_position(5, self.back_data[i])

                # back
                if i < turn_step:
                    self.set_leg_position(
                        1, self.middle_data[left_range[0]:left_range[1]][i])
                    self.set_leg_position(
                        0, self.ahead_data[left_range[0]:left_range[1]][turn_step-1-i]+hang)
                    self.set_leg_position(
                        2, self.back_data[left_range[0]:left_range[1]][turn_step-1-i]+hang)
                time.sleep(time_0)

    def change_state(self):
        if self.state == 1:
            self.state =2
        else:
            self.state=1


    def get_time(self) -> float:
        time_now = vrep.simxGetLastCmdTime(self.client_id)
        return time_now

    def get_body_x_position(self) -> float:
        self.lock.acquire()
        try:
            _, pos = vrep.simxGetObjectPosition(
                self.client_id, self.body, -1, vrep.simx_opmode_blocking)
        finally:
            self.lock.release()
        pos_x = pos[0]
        return pos_x

    def get_body_y_position(self) -> float:
        self.lock.acquire()
        try:
            _, pos = vrep.simxGetObjectPosition(
                self.client_id, self.body, -1, vrep.simx_opmode_blocking)
        finally:
            self.lock.release()
        pos_y = pos[1]
        return pos_y

    def start_simulation(self):
        status = vrep.simxStartSimulation(
            self.client_id, vrep.simx_opmode_oneshot)
        time.sleep(0.5)

    def stop_simulation(self):
        status = vrep.simxStopSimulation(
            self.client_id, vrep.simx_opmode_oneshot)
        time.sleep(0.5)
    
    def pause_simulation(self):
        status = vrep.simxPauseSimulation(
            self.client_id, vrep.simx_opmode_oneshot)
        time.sleep(0.5)

    def get_image(self):
        self.lock.acquire()
        try:
            ret, res, data = vrep.simxGetVisionSensorImage(
                self.client_id, self.vision_sensor, 0, vrep.simx_opmode_buffer)
        finally:
            self.lock.release()
        # print('return code is {}'.format(ret))
        # print('the resolution is {}'.format(res))
        if ret != 0:
            raise Exception('image data get wrong')
        # TODO modified
        data = np.array(data)+128
        # data = 255-(np.array(data)+128)
        data = data.reshape(res[0], res[1], 3)
        data = data.astype(np.float32)
        return data

    def show_speed(self):
        _, spe, _ = vrep.simxGetObjectVelocity(
            self.client_id, self.body, vrep.simx_opmode_oneshot)
        # speed
        print(sqrt(spe[0]**2+spe[1]**2+spe[2]**2),end='\r')


def main():
    # generate_use_data([[10,45],[-25,40],[-50,-10]],50)
    client_id = connect(10)
    rb = Hexapod(client_id)
    rb.start_simulation()
    time.sleep(1)
    rb.step_init()
    while True:
        s_t=time.time()
        rb.one_step_t(0.005)
        rb.show_speed()
        # print(time.time()-s_t)
    # data = rb.get_image()
    # from utils.cv2_util import img_show
    # img_show(data)


if __name__ == "__main__":
    main()
