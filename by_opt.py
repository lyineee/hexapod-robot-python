from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from utils.vrep_util import connect
from hexapod import Hexapod
from regression import generate_use_data_t

import time

from tensorflow import keras

import threading
from queue import Queue
import math
import numpy as np


class ByOpt(object):
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
        # pbounds = {'length': (10, 25), 'st_point_a': (17,27),'st_point_m': (-13, -3),'delay_time': (0.005, 0.015)}
        pbounds = {'mi_0_s': (-5, -40), 'mi_0_e': (5, 40), 'mi_1_s': (0, 35), 'mi_1_m': (30, 50), 'mi_1_e': (0, 35), 'mi_2_s': (-10, -60), 'mi_2_m': (-60, -90), 'mi_2_e': (-10, -60),
                   'ah_0_s': (-40, 5), 'ah_0_e': (5, 40), 'ah_1_s': (0, 35), 'ah_1_m': (30, 50), 'ah_1_e': (0, 35), 'ah_2_s': (-10, -60), 'ah_2_m': (-60, -90), 'ah_2_e': (-10, -60),
                   'ba_0_s': (5, 40), 'ba_0_e': (-40, 5), 'ba_1_s': (0, 35), 'ba_1_m': (30, 50), 'ba_1_e': (0, 35), 'ba_2_s': (-10, -60), 'ba_2_m': (-60, -90), 'ba_2_e': (-10, -60),
                   'delay': (0.130, 0.210)}
        self.optimizer = BayesianOptimization(
            # f=self.black_box,
            f=self.black_box_2,
            pbounds=pbounds,
            random_state=1,)

    def logger_init(self, log_dir="./byopt_logs/logs.json"):
        logger = JSONLogger(path=log_dir)
        self.optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    def start_opt(self, n_iter=200, init_points=50):
        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
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

    def test(self, data):
        # load the data
        ah_data = np.array([[data['ah_0_s'], data['ah_1_s'], data['ah_2_s']], [
                           (data['ah_0_e']+data['ah_0_s'])/2, data['ah_1_m'], data['ah_2_m']],[data['ah_0_e'], data['ah_1_e'], data['ah_2_e']]])
        mi_data=np.array([[data['mi_0_s'], data['mi_1_s'], data['mi_2_s']], [
                           (data['mi_0_e']+data['mi_0_s'])/2, data['mi_1_m'], data['mi_2_m']],[data['mi_0_e'], data['mi_1_e'], data['mi_2_e']]])
        ba_data=np.array([[data['ba_0_e'], data['ba_1_s'], data['ba_2_s']], [
                           (data['ba_0_e']+data['ba_0_s'])/2, data['ba_1_m'], data['ba_2_m']],[data['ba_0_s'], data['ba_1_e'], data['ba_2_e']]])
        # self.rb.set_step_data([ba_data, mi_data, ah_data])
        self.rb.set_step_data([ba_data, mi_data, ah_data])
        self.rb.start_simulation()
        while True:
            self.rb.one_step_t(0.2)
            # self.rb.one_step_t(data['delay'])
            self.rb.show_speed()
            


if __name__ == "__main__":
    tr=ByOpt(ifNN=False)
    # tr.black_box_2(-5.0,11.4296277651623,16.31698153837967,38.84512575870721,22.697966008395372,-10.0,-60.0,-10.0,17.705280260266758,64.11029298688739,30.911962129585323,34.757000163524644,1.090466653651632,-10.0,-60.0,-10.0,6.405387684512668,-55.0,33.12118629201006,48.28824581162664,26.905272497722216,-10.0,-60.0,-10.0,0.18422438495304821)
    data={'ah_0_e': 21.567528489534162, 'ah_0_s': -30.665904245318952, 'ah_1_e': 30.181731733852118, 'ah_1_m': 42.18096679035768, 'ah_1_s':
            19.12090454989839, 'ah_2_e': -10.0, 'ah_2_m': -60.0, 'ah_2_s': -10.0, 'ba_0_e': -39.62079910070136, 'ba_0_s': 13.141020681052858, 'ba_1_e': 4.945834079946826, 'ba_1_m': 47.31038436871174, 'ba_1_s': 34.2966769798406, 'ba_2_e': -10.0, 'ba_2_m': -60.0, 'ba_2_s': -10.0, 'delay': 0.1343843112808652, 'mi_0_e': 15.286912928663536, 'mi_0_s': -5.0, 'mi_1_e': 4.015969321057493, 'mi_1_m': 30.067094254475432, 'mi_1_s': 34.957738957116895, 'mi_2_e': -10.0, 'mi_2_m': -60.0, 'mi_2_s': -10.0}
    # tr.black_box_2(data['mi_0_s'],
    #                data['mi_0_e'],
    #                data['mi_1_s'],
    #                data['mi_1_m'],
    #                data['mi_1_e'],
    #                data['mi_2_s'],
    #                data['mi_2_m'],
    #                data['mi_2_e'],
    #                data['ah_0_s'],
    #                data['ah_0_e'],
    #                data['ah_1_s'],
    #                data['ah_1_m'],
    #                data['ah_1_e'],
    #                data['ah_2_s'],
    #                data['ah_2_m'],
    #                data['ah_2_e'],
    #                data['ba_0_s'],
    #                data['ba_0_e'],
    #                data['ba_1_s'],
    #                data['ba_1_m'],
    #                data['ba_1_e'],
    #                data['ba_2_s'],
    #                data['ba_2_m'],
    #                data['ba_2_e'],
    #                data['delay'])
    # tr.test(data)
    tr.load_logs(['./byopt_logs/logs_tmp.json'])
    tr.logger_init()
    tr.start_opt()
    tr.show_max()
