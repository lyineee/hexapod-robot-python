from hexapod import Hexapod
import vrep
import time
import sys 
import pickle

from math import sin 
from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation
from gaft.analysis import ConsoleOutput,fitness_store

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

indv_template = BinaryIndividual(ranges=[(30,50),(20,30),(-80,-60),(30,50),(20,30),(-80,-60),(30,50),(20,30),(-80,-60)], eps=0.001)
population = Population(indv_template=indv_template, size=50)
population.init()  # Initialize population with individuals.

# Use built-in operators here.
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation, analysis=[ConsoleOutput])


@engine.fitness_register
def fitness(indv):
    data=list(indv.solution)
    fitness=rb.step(data)
    return fitness



@engine.analysis_register
class test(OnTheFlyAnalysis):
    master_only = True
    interval = 1
    def register_step(self, g, population, engine):
        # save population every 10 iteration
        if g%10==0:
            with open('backup-in-{}.pickle'.format(g),'wb') as f:
                pickle.dump(population,f)
        #reset the simulation
        status=-1
        while status!=0:
            status=vrep.simxStopSimulation(client_id,vrep.simx_opmode_blocking)



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

client_id=connect(10)
rb=Hexapod(client_id)

if __name__ == "__main__":
    # client_id=connect(10)
    # rb=Hexapod(client_id)
    # vrep.simxStartSimulation(client_id,vrep.simx_opmode_blocking)
    rb.init()
    engine.run(ng=100)
