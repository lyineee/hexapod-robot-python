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




class ByOpt(object):
    def __init__(self):
        #val init
        self.log_dir=''
        #vrep init
        self.client_id=connect(5)
        self.rb=Hexapod(self.client_id)
        #NN init
        self.model=[]
        pre = ['./data/NN_ahead', './data/NN_middle', './data/NN_back']
        for i in range(3):
            self.model.append(keras.models.load_model('{}.h5'.format(pre[i])))
        #byopt init
        pbounds = {'length': (10, 25), 'st_point_a': (17,27),'st_point_m': (-13, -3),'delay_time': (0.005, 0.015)}
        self.optimizer = BayesianOptimization(
            f=self.black_box,
            pbounds=pbounds,
            random_state=1,)
    def logger_init(self,log_dir="./byopt_logs/logs.json"):
        logger = JSONLogger(path=log_dir)
        self.optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    def start_opt(self,n_iter=100,init_points=10):
        self.optimizer.maximize(
                init_points=init_points,
                n_iter=n_iter,
            )   
    
    def load_logs(self,log_dir=["./byopt_logs/logs.json"]):
        load_logs(self.optimizer,logs=log_dir)

    def show_max(self):
        max=self.optimizer.max
        print(max)
        return max


    def black_box(self,length,st_point_a,st_point_m,delay_time):
        self.rb.start_simulation()
        generate_use_data_t([[st_point_a,st_point_a-length],[st_point_m+length,st_point_m],[-1*st_point_a+length,-1*st_point_a]],25,self.model)
        q=Queue()
        th=threading.Thread(target=self.rb.keep_step,name='step_th',args=(delay_time,q))
        time.sleep(1)
        start_time=time.time()
        th.start() #start walk
        q.put('start')
        while time.time()-start_time<5:
            time.sleep(0.3)
        q.put('end')
        self.rb.pause_simulation()
        time.sleep(0.4)
        position=self.rb.get_body_x_position()
        self.rb.stop_simulation()
        print(position)
        th.join()
        if math.isnan(position):
            position=4
        return position

if __name__ == "__main__":
    tr=ByOpt()
    tr.load_logs(['./byopt_logs/logs_tmp.json'])
    tr.logger_init()
    tr.start_opt()
    tr.show_max()        
