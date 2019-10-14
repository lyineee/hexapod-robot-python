from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from utils.vrep_util import connect
from hexapod import Hexapod
from regression import generate_use_data_t
from image_train import refresh_state,test_method

import time

from tensorflow import keras

import threading
from queue import Queue
import math
import numpy as np


class ByOpt(object):
    test=0
    count=1
    def __init__(self, ifNN=True):
        # val init
        self.log_dir = ''
        # vrep init
        self.client_id = connect(5)
        self.rb = Hexapod(self.client_id)
        if ifNN:
            # NN init
            self.model = []
            pre = ['./data/NN_ahead', './data/NN_middle', './data/NN_back']
            for i in range(3):
                self.model.append(
                    keras.models.load_model('{}.h5'.format(pre[i])))
        # byopt init
        pbounds = {'length': (10, 25), 'st_point_a': (17,27),'st_point_m': (-13, -3),'delay_time': (0.005, 0.015)}
        pbounds_2 = {'mi_0_s': (-5, -40), 'mi_0_e': (5, 40), 'mi_1_s': (0, 35), 'mi_1_m': (30, 50), 'mi_1_e': (0, 35), 'mi_2_s': (-10, -60), 'mi_2_m': (-60, -90), 'mi_2_e': (-10, -60),
                   'ah_0_s': (-40, 5), 'ah_0_e': (5, 40), 'ah_1_s': (0, 35), 'ah_1_m': (30, 50), 'ah_1_e': (0, 35), 'ah_2_s': (-10, -60), 'ah_2_m': (-60, -90), 'ah_2_e': (-10, -60),
                   'ba_0_s': (5, 40), 'ba_0_e': (-40, 5), 'ba_1_s': (0, 35), 'ba_1_m': (30, 50), 'ba_1_e': (0, 35), 'ba_2_s': (-10, -60), 'ba_2_m': (-60, -90), 'ba_2_e': (-10, -60),
                   'delay': (0.130, 0.210)}
        pbounds_3 = {'mi_0_s': (-5, -40),  'mi_0_e': (5, 40), 'mi_1_s': (0, 35), 'mi_1_e': (0, 35), 'mi_2_s': (-10, -60), 'mi_2_e': (-10, -60),
                   'ah_0_s': (-40, 5), 'ah_0_e': (5, 40), 'ah_1_s': (0, 35), 'ah_1_e': (0, 35), 'ah_2_s': (-10, -60), 'ah_2_e': (-10, -60),
                   'ba_0_s': (5, 40), 'ba_0_e': (-40, 5), 'ba_1_s': (0, 35), 'ba_1_e': (0, 35), 'ba_2_s': (-10, -60), 'ba_2_e': (-10, -60),
                   'delay': (0.130, 0.210)}
        pbounds_4 = {'mi_0_s': (20, -45),  'mi_0_e': (-20, 45), 'mi_1_s': (-20, 45), 'mi_1_e': (-20, 45), 'mi_2_s': (10, -80), 'mi_2_e': (10, -80),
                   'ah_0_s': (-45, 30), 'ah_0_e': (-30, 45), 'ah_1_s': (-20, 45), 'ah_1_e': (-20, 45), 'ah_2_s': (10, -80), 'ah_2_e': (10, -80),
                   'delay': (0.130, 0.210)}
        pbounds_5={'ahead_1_a':(-1,1),'ahead_2_a':(-1,1),'ahead_3_a':(-1,1),'ahead_1_b':(-1,1),'ahead_2_b':(-1,1),'ahead_3_b':(-1,1),'middle_1_a':(-1,1),'middle_2_a':(-1,1),'middle_3_a':(-1,1),'middle_1_b':(-1,1),'middle_2_b':(-1,1),'middle_3_b':(-1,1),'delay':(0.02,0.09)}
        # pbounds_6={'ahead_a':(0.45,0.65),'ahead_b':(,1),'middle_a':(-1,1),'middle_b':(-1,1)}
        # pbounds_6={'ahead_a':(-1,1),'ahead_b':(-1,1),'middle_a':(-1,1),'middle_b':(-1,1)}
        pbounds_6={'ahead_a':(0.63,0.45),'ahead_b':(-0.95,-0.74),'middle_a':(0.45,0.76),'middle_b':(-0.4,-0.2),'delay':(0.025,0.035)}
        pbounds_7={'ahead_a':(-1,-0.9),'ahead_b':(0.35,0.55),'middle_a':(-0.7,-0.6),'middle_b':(-0.03,0.2)}

        self.optimizer = BayesianOptimization(
            # f=self.black_box,
            f=self.black_box_7,
            pbounds=pbounds_6,
            verbose=2,
            random_state=1,)

        # self.q=Queue()
        # self.cap_th = threading.Thread(target=refresh_state, args=(self.rb, 'result.h5',self.q))
        # self.cap_th.start()
        # self.q.get()

    def logger_init(self, log_dir="./byopt_logs/logs.json"):
        logger = JSONLogger(path=log_dir)
        self.optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    def start_opt(self, n_iter=420, init_points=70):
        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
            # kappa=0.5
        )

    def load_logs(self, log_dir=["./byopt_logs/logs.json"]):
        load_logs(self.optimizer, logs=log_dir)

    def show_max(self):
        max = self.optimizer.max
        print(max)
        return max

    def black_box(self, length, st_point_a, st_point_m, delay_time):
        self.rb.start_simulation()
        generate_use_data_t([[st_point_a, st_point_a-length], [st_point_m+length,
                                                               st_point_m], [-1*st_point_a+length, -1*st_point_a]], 25, self.model)
        q = Queue()
        th = threading.Thread(target=self.rb.keep_step,
                              name='step_th', args=(delay_time, q))
        time.sleep(1)
        start_time = time.time()
        th.start()  # start walk
        q.put('start')
        if self.test==1:
            while True:
                time.sleep(0.3)
                self.rb.show_speed()
        while time.time()-start_time < 5:
            time.sleep(0.3)
        q.put('end')
        self.rb.pause_simulation()
        time.sleep(0.4)
        position = self.rb.get_body_x_position()
        self.rb.stop_simulation()
        print(position)
        th.join()
        if math.isnan(position):
            position = 4
        return position

    def black_box_2(self, mi_0_s, mi_0_e, mi_1_s, mi_1_m, mi_1_e, mi_2_s, mi_2_m, mi_2_e, ah_0_s, ah_0_e, ah_1_s, ah_1_m, ah_1_e, ah_2_s, ah_2_m, ah_2_e, ba_0_s, ba_0_e, ba_1_s, ba_1_m, ba_1_e, ba_2_s, ba_2_m, ba_2_e, delay):
        # load the data
        ah_data = np.array([[ah_0_s, ah_1_s, ah_2_s], [
                           (ah_0_e+ah_0_s)/2, ah_1_m, ah_2_m], [ah_0_e, ah_1_e, ah_2_e]])
        mi_data = np.array([[mi_0_s, mi_1_s, mi_2_s], [
                           (mi_0_e+mi_0_s)/2, mi_1_m, mi_2_m], [mi_0_e, mi_1_e, mi_2_e]])
        ba_data = np.array([[ba_0_e, ba_1_s, ba_2_s], [
                           (ba_0_e+ba_0_s)/2, ba_1_m, ba_2_m], [ba_0_s, ba_1_e, ba_2_e]])
        self.rb.set_step_data([ba_data, mi_data, ah_data])

        self.rb.start_simulation()
        q = Queue()
        th = threading.Thread(target=self.rb.keep_step,
                              name='step_th', args=(delay, q))
        time.sleep(1)
        start_time = time.time()
        th.start()  # start walk
        q.put('start')
        if self.test==1:
            while True:
                time.sleep(0.3)
                self.rb.show_speed()
        while time.time()-start_time < 15:
            time.sleep(0.3)
        q.put('end')
        self.rb.pause_simulation()
        time.sleep(0.4)
        position = self.rb.get_body_x_position()
        self.rb.stop_simulation()
        print(position)
        th.join()
        if math.isnan(position):
            position = 0
        return position

    def black_box_3(self, mi_0_s, mi_0_e, mi_1_s, mi_1_e, mi_2_s, mi_2_e, ah_0_s, ah_0_e, ah_1_s,   ah_1_e, ah_2_s,   ah_2_e, ba_0_s, ba_0_e, ba_1_s,   ba_1_e, ba_2_s,   ba_2_e, delay):
        # load the data
        ah_data = np.array(
            [[ah_0_s, ah_1_s, ah_2_s],  [ah_0_e, ah_1_e, ah_2_e]])
        mi_data = np.array(
            [[mi_0_s, mi_1_s, mi_2_s], [mi_0_e, mi_1_e, mi_2_e]])
        ba_data = np.array(
            [[ba_0_e, ba_1_s, ba_2_s], [ba_0_s, ba_1_e, ba_2_e]])
        self.rb.set_step_data([ba_data, mi_data, ah_data])

        self.rb.start_simulation()
        q = Queue()
        th = threading.Thread(target=self.rb.keep_step,
                              name='step_th', args=(delay, q))
        time.sleep(1)
        start_time = time.time()
        th.start()  # start walk
        q.put('start')
        if self.test==1:
            while True:
                time.sleep(0.3)
                self.rb.show_speed()
        while time.time()-start_time < 5:
            time.sleep(0.3)
        q.put('end')
        self.rb.pause_simulation()
        time.sleep(0.4)
        position = self.rb.get_body_x_position()
        self.rb.stop_simulation()
        print(position)
        th.join()
        if math.isnan(position):
            position = 0
        return position


    def black_box_4(self, mi_0_s, mi_0_e, mi_1_s, mi_1_e, mi_2_s, mi_2_e, ah_0_s, ah_0_e, ah_1_s,ah_1_e, ah_2_s,ah_2_e,delay):
        # load the data
        ah_data = np.array(
            [[ah_0_s, ah_1_s, ah_2_s],  [ah_0_e, ah_1_e, ah_2_e]])
        mi_data = np.array(
            [[mi_0_s, mi_1_s, mi_2_s], [mi_0_e, mi_1_e, mi_2_e]])
        self.rb.set_step_data([ah_data, mi_data, ah_data])

        self.rb.start_simulation()
        q = Queue()
        th = threading.Thread(target=self.rb.keep_step,
                              name='step_th', args=(delay, q))
        time.sleep(1)
        start_time = time.time()
        th.start()  # start walk
        q.put('start')
        if self.test==1:
            while True:
                time.sleep(0.3)
                self.rb.show_speed()
        while time.time()-start_time < 15:
            time.sleep(0.3)
        q.put('end')
        self.rb.pause_simulation()
        time.sleep(0.4)
        position = self.rb.get_body_x_position()
        self.rb.stop_simulation()
        print(position)
        th.join()
        if math.isnan(position):
            position = 0
        return position

    def black_box_5(self,ahead_1_a,ahead_2_a,ahead_3_a,ahead_1_b,ahead_2_b,ahead_3_b,middle_1_a,middle_2_a,middle_3_a,middle_1_b,middle_2_b,middle_3_b,delay):
        ahead_leg_a=np.array([ahead_1_a,ahead_2_a,ahead_3_a])
        ahead_leg_b=np.array([ahead_1_b,ahead_2_b,ahead_3_b])

        middle_leg_a=np.array([middle_1_a,middle_2_a,middle_3_a])
        middle_leg_b=np.array([middle_1_b,middle_2_b,middle_3_b])

        ahead_leg=ahead_leg_a*np.sin(np.linspace(0,np.pi/2,5).reshape(5,1))+ahead_leg_b*np.cos(np.linspace(0,np.pi/2,5).reshape(5,1))
        middle_leg=middle_leg_a*np.sin(np.linspace(0,np.pi/2,5).reshape(5,1))+middle_leg_b*np.cos(np.linspace(0,np.pi/2,5).reshape(5,1))

        ah_data = np.stack(ahead_leg)*60
        mi_data = np.stack(middle_leg)*60

        self.rb.set_step_data([ah_data, mi_data, ah_data])

        self.rb.start_simulation()
        q = Queue()
        th = threading.Thread(target=self.rb.keep_step,
                              name='step_th', args=(delay, q))
        time.sleep(1)
        start_time = time.time()
        th.start()  # start walk
        q.put('start')
        if self.test==1:
            while True:
                time.sleep(0.3)
                self.rb.show_speed()
        while time.time()-start_time < 15:
            time.sleep(0.3)
        q.put('end')
        self.rb.pause_simulation()
        time.sleep(0.4)
        position = self.rb.get_body_x_position()
        self.rb.stop_simulation()
        print(position)
        th.join()
        if math.isnan(position):
            position = 0
        return position

    def black_box_6(self,ahead_a,ahead_b,middle_a,middle_b,delay):
        delay=0.02573650763940472
        ahead_leg_a=np.array([0.5446098736046021,-0.9474735404250036,0.2553768404632672])
        ahead_leg_b=np.array([-0.835653908906631,0.39316038539104503,-0.38129175361998874])

        middle_leg_a=np.array([ 0.665468853686944,-0.707220332164328,0.03495425595681743])
        middle_leg_b=np.array([ -0.4075119658229989,-0.025838715404019164,0.15507920849877677])

        ahead_leg=ahead_leg_a*np.sin(np.linspace(0,np.pi/2,5).reshape(5,1))+ahead_leg_b*np.cos(np.linspace(0,np.pi/2,5).reshape(5,1))
        middle_leg=middle_leg_a*np.sin(np.linspace(0,np.pi/2,5).reshape(5,1))+middle_leg_b*np.cos(np.linspace(0,np.pi/2,5).reshape(5,1))


        ah_data = np.stack(ahead_leg)*60
        mi_data = np.stack(middle_leg)*60

        # ah_data[:,0]=np.linspace(-10,10,5)
        # mi_data[:,0]=np.linspace(-10,10,5)

        self.rb.set_step_data([ah_data, mi_data, ah_data])

        self.rb.start_simulation()
        q = Queue()
        th = threading.Thread(target=self.rb.keep_step,
                              name='step_th', args=(delay, q))
        time.sleep(1)
        start_time = time.time()
        th.start()  # start walk
        q.put('start')
        if self.test==1:
            while True:
                time.sleep(0.3)
                self.rb.show_speed()
        while time.time()-start_time < 15:
            time.sleep(0.3)
        q.put('end')
        self.rb.pause_simulation()
        time.sleep(0.4)
        position = self.rb.get_body_x_position()
        self.rb.stop_simulation()
        print(position)
        th.join()
        if math.isnan(position):
            position = 0
        return position


    def black_box_7(self,ahead_a,ahead_b,middle_a,middle_b,delay):
        #ahead_data
        # delay=0.02573650763940472
        ahead_leg_a=np.array([0.5446098736046021,-0.9474735404250036,0.2553768404632672])
        ahead_leg_b=np.array([-0.835653908906631,0.39316038539104503,-0.38129175361998874])

        middle_leg_a=np.array([ 0.665468853686944,-0.707220332164328,0.03495425595681743])
        middle_leg_b=np.array([ -0.4075119658229989,-0.025838715404019164,0.15507920849877677])

        ahead_leg=ahead_leg_a*np.sin(np.linspace(0,np.pi/2,5).reshape(5,1))+ahead_leg_b*np.cos(np.linspace(0,np.pi/2,5).reshape(5,1))
        middle_leg=middle_leg_a*np.sin(np.linspace(0,np.pi/2,5).reshape(5,1))+middle_leg_b*np.cos(np.linspace(0,np.pi/2,5).reshape(5,1))

        ah_data = np.stack(ahead_leg)*60
        mi_data = np.stack(middle_leg)*60

        # self.rb.set_step_data([ah_data, mi_data, ah_data])

        #turn data
        # delay=0.04866337735403608
        ahead_turn_leg_a=np.array([0,-0.9474735404250036,0.2553768404632672])
        ahead_turn_leg_b=np.array([0,0.39316038539104503,-0.38129175361998874])

        middle_turn_leg_a=np.array([ 0,-0.707220332164328,0.03495425595681743])
        middle_turn_leg_b=np.array([ 0,-0.025838715404019164,0.15507920849877677])

        # ahead_turn_leg_a=np.array([ahead_a,0,0])
        # ahead_turn_leg_b=np.array([ahead_b,0,0])

        # middle_turn_leg_a=np.array([middle_a,0,0])
        # middle_turn_leg_b=np.array([middle_b,0,0])

        ahead_turn_leg=ahead_turn_leg_a*np.sin(np.linspace(0,np.pi/2,5).reshape(5,1))+ahead_turn_leg_b*np.cos(np.linspace(0,np.pi/2,5).reshape(5,1))
        middle_turn_leg=middle_turn_leg_a*np.sin(np.linspace(0,np.pi/2,5).reshape(5,1))+middle_turn_leg_b*np.cos(np.linspace(0,np.pi/2,5).reshape(5,1))

        ah_turn_data = np.stack(ahead_turn_leg)*60
        mi_turn_data = np.stack(middle_turn_leg)*60

        ah_turn_data[:,0]=np.linspace(-3,3,5)
        mi_turn_data[:,0]=np.linspace(-3,3,5)
        #TODO ok

        self.rb.set_step_data([ah_data, mi_data, ah_data],[ah_turn_data, mi_turn_data, ah_turn_data],delay)

        self.rb.start_simulation()
        q = Queue()
        th = threading.Thread(target=self.rb.keep_step,
                              name='step_th', args=(delay, q,self.rb.right))
        # time.sleep(1)
        # use_time=test_method(self.rb)

        # on_the_air_data=[]
        time_start=time.time()
        th.start()
        q.put('start')
        if self.test==1:
            while True:
                time.sleep(0.3)
                self.rb.show_speed()
        while time.time()-time_start < 6:
            time.sleep(0.3)
            # on_the_air_data.append([self.rb.get_body_x_position(),self.rb.get_body_y_position()])
        q.put('end')

        self.rb.pause_simulation()

        # on_the_air_data=np.array(on_the_air_data)
        # on_the_air_result=(np.sqrt(on_the_air_data[:,1]**2+(on_the_air_data[:,0]-1.875)**2)-1.875)**2 #Standard Deviation 
        # std=(-1)*np.std(on_the_air_result,ddof=1) #reverse the std

        time.sleep(0.4)
        x=self.rb.get_body_x_position()
        y=(-1)*self.rb.get_body_y_position()
        self.rb.stop_simulation()
        if y<0:
            return -2
        arg1=(-1)*(x-math.sqrt(1.875**2-(y-1.875)**2))**2

        # deg=np.degrees(np.arctan(x/(y-1.875)))
        # # if x>0 and y-1.875<0:
        # #     deg=180+deg
        # # arg1=(deg/180)*10
        # length_data=on_the_air_data
        # length_data[1:]=on_the_air_data[1:]-on_the_air_data[:-1]
        # length=sum(np.sqrt(length_data[:,0]**2+length_data[:,1]**2))
        # speed=length/6
        result=3*arg1+y
        print('{}: {}'.format(self.count,result))
        self.count+=1
        return result
        
    def test_data(self,data_opt,black_box_func):
        # load the data
        # ah_data = np.array([[data['ah_0_s'], data['ah_1_s'], data['ah_2_s']], [
        #                    (data['ah_0_e']+data['ah_0_s'])/2, data['ah_1_m'], data['ah_2_m']], [data['ah_0_e'], data['ah_1_e'], data['ah_2_e']]])
        # mi_data = np.array([[data['mi_0_s'], data['mi_1_s'], data['mi_2_s']], [
        #                    (data['mi_0_e']+data['mi_0_s'])/2, data['mi_1_m'], data['mi_2_m']], [data['mi_0_e'], data['mi_1_e'], data['mi_2_e']]])
        # ba_data = np.array([[data['ba_0_e'], data['ba_1_s'], data['ba_2_s']], [
        #                    (data['ba_0_e']+data['ba_0_s'])/2, data['ba_1_m'], data['ba_2_m']], [data['ba_0_s'], data['ba_1_e'], data['ba_2_e']]])
        # self.rb.set_step_data([ba_data, mi_data, ah_data])

        ah_data = np.array([[data['ah_0_s'], data['ah_1_s'], data['ah_2_s']],  [
                           (data['ah_0_e']+data['ah_0_s'])/2,(data['ah_1_s']+data['ah_1_e'])/2,(data['ah_2_e']+data['ah_2_s'])/2],[data['ah_0_e'], data['ah_1_e'], data['ah_2_e']]])
        mi_data = np.array([[data['mi_0_s'], data['mi_1_s'], data['mi_2_s']], [
                           (data['mi_0_e']+data['mi_0_s'])/2,(data['mi_1_s']+data['mi_1_e'])/2,(data['mi_2_e']+data['mi_2_s'])/2],[data['mi_0_e'], data['mi_1_e'], data['mi_2_e']]])
        ba_data = np.array([[data['ba_0_e'], data['ba_1_s'], data['ba_2_s']],[
                           (data['ba_0_e']+data['ba_0_s'])/2,(data['ba_1_s']+data['ba_1_e'])/2,(data['ba_2_e']+data['ba_2_s'])/2],[data['ba_0_s'], data['ba_1_e'], data['ba_2_e']]])
        # ba_data=ah_data
        # self.rb.set_step_data([ba_data, mi_data, ah_data])
        # self.rb.start_simulation()
        # while True:
        #     # self.rb.one_step_t(0.2)
        #     self.rb.one_step_t(data['delay'])
        #     self.rb.show_speed()
        self.test=1
        black_box_func(**data_opt)
        self.test=0


if __name__ == "__main__":
    tr = ByOpt(ifNN=False)
    # tr.black_box_2(-5.0,11.4296277651623,16.31698153837967,38.84512575870721,22.697966008395372,-10.0,-60.0,-10.0,17.705280260266758,64.11029298688739,30.911962129585323,34.757000163524644,1.090466653651632,-10.0,-60.0,-10.0,6.405387684512668,-55.0,33.12118629201006,48.28824581162664,26.905272497722216,-10.0,-60.0,-10.0,0.18422438495304821)

    #TODO keep this data, might be useful
    #black_box_3
    data_1= {"ah_0_e": 39.60156540443571, "ah_0_s": -13.911465133939139, "ah_1_e": 13.304941041824264, "ah_1_s": 19.283187669126388, "ah_2_e": -47.266721545325105, "ah_2_s": -43.46164467265923, "ba_0_e": -28.078619905173575, "ba_0_s": 7.321719204995455, "ba_1_e": 12.95294692699372, "ba_1_s": 22.040112745754758, "ba_2_e": -20.50870049574198, "ba_2_s": -47.637777686940694, "delay": 0.1353229185083292, "mi_0_e": 14.111028450248934, "mi_0_s": -33.16640973101709, "mi_1_e": 6.770199891816471, "mi_1_s": 22.381130830797904, "mi_2_e": -36.23351545618668, "mi_2_s": -56.240398519967535}
    #black_box_4
    data_2={'ah_0_e': 1.27665035269305, 'ah_0_s': 9.024337008161858, 'ah_1_e': -19.992565636872584, 'ah_1_s': -0.34838277893041436, 'ah_2_e': -3.2080301735401733, 'ah_2_s': 1.6895264708081985, 'delay': 0.14490081691021367, 'mi_0_e': 2.4614472577981026, 'mi_0_s': -5.789885824993547, 'mi_1_e': 15.0230877102182, 'mi_1_s': 7.247643436214162, 'mi_2_e': -51.669755035708356, 'mi_2_s': -8.400702475836567}

    #black_box_2
    data_4={'ah_0_e': 26.803138261623868, 'ah_0_s': -30.782681224284914, 'ah_1_e': 13.676852061995204, 'ah_1_m': 31.415106182836055, 'ah_1_s': 28.57090734303387, 'ah_2_e': -10.0, 'ah_2_m': -60.0, 'ah_2_s': -10.0, 'ba_0_e': -39.69888705051125, 'ba_0_s': 6.529029911177014, 'ba_1_e': 30.426046787466426, 'ba_1_m': 47.99466656762513, 'ba_1_s': 12.406299853720054, 'ba_2_e': -10.0, 'ba_2_m': -60.0, 'ba_2_s': -10.0, 'delay': 0.18741691445922454, 'mi_0_e': 31.48478529870462, 'mi_0_s': -5.0, 'mi_1_e': 13.961845174906234, 'mi_1_m': 31.062302868011056, 'mi_1_s': 15.560760335829798, 'mi_2_e': -10.0, 'mi_2_m': -60.0, 'mi_2_s': -10.0}
    data = {'ah_0_e': 21.567528489534162, 'ah_0_s': -30.665904245318952, 'ah_1_e': 30.181731733852118, 'ah_1_m': 42.18096679035768, 'ah_1_s':19.12090454989839, 'ah_2_e': -10.0, 'ah_2_m': -60.0, 'ah_2_s': -10.0, 'ba_0_e': -39.62079910070136, 'ba_0_s': 13.141020681052858, 'ba_1_e': 4.945834079946826, 'ba_1_m': 47.31038436871174, 'ba_1_s': 34.2966769798406, 'ba_2_e': -10.0, 'ba_2_m': -60.0, 'ba_2_s': -10.0, 'delay': 0.1343843112808652, 'mi_0_e': 15.286912928663536, 'mi_0_s': -5.0, 'mi_1_e': 4.015969321057493, 'mi_1_m': 30.067094254475432, 'mi_1_s': 34.957738957116895, 'mi_2_e': -10.0, 'mi_2_m': -60.0, 'mi_2_s': -10.0}
    #black_box_5
    data_5={'ahead_1_a': -0.11787471891040746,'ahead_1_b': -0.35979252290763664,'ahead_2_a': -0.8051121723991821,'ahead_2_b': 0.2673552110494355,'ahead_3_a': -0.17858107322978412,'ahead_3_b': 0.6018015322098393,'delay': 0.03391338047339743,'middle_1_a': 0.5914147012457913,'middle_1_b': 0.11766459295904363,'middle_2_a': -0.4261450552636197,'middle_2_b': 0.6619504757239809,'middle_3_a': -0.21450701395992966,'middle_3_b': -0.8552356097886922}
    

    #??? -10 black_box_6
    data_6={"ahead_2_a": -0.793547986844716, "ahead_2_b": -0.10421294764818967, "middle_2_a": 0.8171910061861911, "middle_2_b": -0.412771703252641}
    data_7={'ahead_1_a': 0.5446098736046021, 'ahead_1_b': -0.835653908906631, 'delay': 0.02573650763940472, 'middle_1_a': 0.665468853686944, 'middle_1_b': -0.4075119658229989}
    data_7={'ahead_1_a': 0.5446098736046021, 'ahead_1_b': -0.835653908906631, 'delay': 0.03573650763940472, 'middle_1_a': 0.665468853686944, 'middle_1_b': -0.4075119658229989}
    data_7={'ahead_a': 0.5446098736046021, 'ahead_b': -0.835653908906631, 'delay': 0.03573650763940472, 'middle_a': 0.665468853686944, 'middle_b': -0.4075119658229989}

    # tr.test(data_7,tr.black_box_6)
    #black_box_7 
    data_8={'ahead_a': 0.7973475208988559, 'ahead_b': -0.9925179072490438, 'delay': 0.04676828436230832, 'middle_a': 0.7606830742221407, 'middle_b': -0.3833777503252276}
    data_9 = {'ahead_a': 0.7855599514433688, 
              'ahead_b': -1.0,
            #   'delay': 0.04866337735403608,
              'middle_a': 0.7379049669844442,
              'middle_b': -0.36132918484862947}#use this data data[-2] this is leg_0 data
    data_10 = {'ahead_a': 0.8139253072428453,
               'ahead_b': -0.9630527989237955,
               'delay': 0.04795460569137124,
               'middle_a': 0.7961344428871164,
               'middle_b': -0.3961634973231694}

    #new one 
    data_11 ={'ahead_a': 0.63,                                                                                'ahead_b': -0.8667278955654526,
  'delay': 0.03356217420325619,
  'middle_a': 0.6123380307461854,
  'middle_b': -0.26129529963319836}



    tr.test_data(data_11,tr.black_box_7)
    # tr.load_logs(['./byopt_logs/logs_tmp.json'])
    # tr.logger_init()
    # tr.start_opt(100,30)
    # # tr.start_opt(450,80)
    # tr.show_max()

#挑一部分参数出来多次优化？减少优化参数的个数
