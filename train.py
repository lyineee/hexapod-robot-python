from hexapod import Hexapod
import vrep
import time
import sys 
import pickle
import numpy as np 
import pickle

from tensorflow import keras

from math import sin 
from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation
from gaft.analysis import ConsoleOutput,fitness_store

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

indv_template = BinaryIndividual(ranges=[(-10,35),(40,60),(-40,-15),(15,50),(-70,-30),(0,20),(0.001,0.01)], eps=0.001)
population = Population(indv_template=indv_template, size=16)
population.init()  # Initialize population with individuals.
# with open('./backup-in-10.pickle','rb') as f:
# population=pickle.load(f)

# Use built-in operators here.
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation, analysis=[ConsoleOutput])


@engine.fitness_register
def fitness(indv):
    rb.start_simulation()
    # time.sleep(0.5)
    rb.init()
    run_time=20000
    data=list(indv.solution)
    rb.set_step_data(stepData.gen_data([data[:2],data[2:4],data[4:6]]))
    # start_time=rb.get_time()
    # while rb.get_time()-start_time<run_time:
    rb.step(data[6],run_time)
    # print(start_time)
    fitness=rb.get_body_x_position()
    rb.stop_simulation()
    time.sleep(1)
    return fitness



# @engine.analysis_register
# class test(OnTheFlyAnalysis):
#     master_only = True
#     interval = 1
#     def register_step(self, g, population, engine):
#         # save population every 10 iteration
#         if g%10==0:
#             with open('backup-in-{}.pickle'.format(g),'wb') as f:
#                 pickle.dump(population,f)
#         L=50
#         N=50
#         print("{{{0}>{1}}} {2}%".format('='*round(g*L/N),
#                                             '.'*round((N-g)*L/N), round(g*100/N)), end="\r")



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

class Data(object):
    def __init__(self):
        self.model=[]
        pre=['./data/NN_ahead','./data/NN_middle','./data/NN_back']
        for i in range(3):
            self.model.append(keras.models.load_model('{}.h5'.format(pre[i])))
    def gen_data(self,sample_range,sample_rate=50):
        self.data=[]
        for i in range(3):
            model=self.model[i]
            sampling_point=np.linspace(min(sample_range[i]),max(sample_range[i]),sample_rate)
            sampling_point=sampling_point.reshape(sampling_point.shape[0],1)
            predict=model.predict(sampling_point)
            sampling_point= np.around(sampling_point,decimals=1)
            predict= np.around(predict,decimals=1)
            result=np.concatenate([sampling_point,predict],axis=1)
            self.data.append(result)
        return self.data



client_id=connect(10)
rb=Hexapod(client_id)

stepData=Data()

if __name__ == "__main__":
    engine.run(ng=50)
