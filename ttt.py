from leg import Leg
from RL_brain import DeepQNetwork
import vrep
import sys
import time


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        env.reset()
        observation,reward,done,info =env.step()
        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)
            action*=100

            # RL take action and get next observation and reward
            observation_,reward,done,info = env.step(action)
            print(observation_,reward)
            action=list(action[0])
            RL.store_transition(observation, action, reward, observation_)

            if (step > 15) and (step % 3 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')

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
    # maze game
    env = Leg(connect(10))
    RL = DeepQNetwork(2, 1,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=20,
                      memory_size=100,
                      # output_graph=True
                      )
    # env.after(100, run_maze)
    # env.mainloop()
    run_maze()
    RL.plot_cost()