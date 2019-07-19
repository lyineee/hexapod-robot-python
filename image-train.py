from hexapod import Hexapod
import vrep
import cv2
import numpy as np
import sys
import time

import tensorflow as tf
from tensorflow import keras

import os
import threading

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

def cap_loop(ver,rb,sleep_time):
    global lock
    cap=ImgCap(ver)
    while True:
        lock.acquire()
        try:
            cap.capture(rb.get_image(),label)
        except Exception as e:
            time.sleep(sleep_time)
            continue
        finally:
            lock.release()
        time.sleep(sleep_time)

def sample_data():
    client_id = connect(10)
    rb = Hexapod(client_id)
    vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)
    time.sleep(1)
    rb.step_init()
    position_state = 0

    #creat thread
    global label,lock
    label=0
    lock=threading.Lock()
    cap_th=threading.Thread(target=cap_loop,args=(5,rb,0.05))
    cap_th.start()

    while True:
        lock.acquire()
        try:
            errorcode, position = vrep.simxGetObjectPosition(
                client_id, rb.body, -1, vrep.simx_opmode_blocking)
        finally:
            lock.release()

        if position[0] < 4.6 and position_state == 0:

            rb.one_step(0.005)
            label=0

        elif position[0] >= 4.6 and position[1] < 1.3:
            if position_state == 0:
                position_state = 1
            
            rb.go_left()
            label=1

        elif position[0] > 5.6 and 2.7 > position[1] >= 1.3:  
            if position_state == 1:
                position_state = 2
            rb.go_left()
            label=1

        elif 2.8 <= position[0] < 5.5 and 4.2 > position[1] >= 2.4:
            if position_state == 2:
                position_state = 3
            rb.one_step(0.005)
            label=0

        elif position[0] < 2.8 and 4.2 > position[1] >= 2.7:
            if position_state == 3:
                position_state = 4
            rb.go_right()
            label=2

        elif position[0] < 2.8 and 6.7 > position[1] >= 4.2:
            if position_state == 4:
                position_state = 5
            rb.go_right()
            label=2

        else:
            rb.one_step(0.005)
            label=0


class ImgCap(object):
    def __init__(self,ver):
        self.index=0
        self.ver=ver
        self.create_dir('./image/{}'.format(self.ver))
    
    def capture(self,data, label):
        data=cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
        cv2.imwrite('./image/{0}/image_{1}_{2}.jpg'.format(self.ver,self.index, label), data)
        self.index+=1

    def create_dir(self,dirs):
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        else:
            file_list=os.listdir('./image/{}'.format(self.ver))
            if not file_list==[]:
                for name in file_list:
                    os.remove('./image/{0}/{1}'.format(self.ver,name))


def build_model(X):
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=X.shape[1:]),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        # keras.layers.Dense(64, activation='softmax'),
        keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])
    return model


def train():
    # get data
    label, data = get_data()
    # build model
    model = build_model(data)
    # train
    tbCallBack = keras.callbacks.TensorBoard(log_dir='.\logs')
    model.fit(data, label, batch_size=32, epochs=3,
              validation_split=0.1, callbacks=[tbCallBack])
    # save model
    model.save('result.h5')


def get_data():
    # get file name
    img_size = (28, 28)
    img_path = './image/'
    label = []
    file_name_list = os.listdir(img_path)
    data = np.zeros((1,)+img_size, dtype=np.float32)
    for file_name in file_name_list:
        # data
        file_name = img_path+file_name
        img = cv2.imread(file_name)
        img_resize = cv2.resize(img, img_size)
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_array = np.array(img_gray)
        img_array = img_array.reshape((1,)+img_size)
        data = np.concatenate([data, img_array], axis=0)
        # label
        label_name = int(os.path.splitext(file_name)[0].split('_')[-1])
        label.append(label_name)

    data = np.delete(data, 0, axis=0)
    label = np.array(label)
    return label, data


if __name__ == "__main__":
    # for i in range(50):
    #     data=rb.get_image()
    #     # img=cv2.fromarray(data)
    #     for j in range(64*64*3):
    #         tmp=data.reshape(-1)[j]
    #         if tmp >128:
    #             data.reshape(-1)[j]=255
    #         else:
    #             data.reshape(-1)[j]=0
    # train()
    # cap=ImgCap(0)
    sample_data()
    pass
