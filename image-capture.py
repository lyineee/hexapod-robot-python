from hexapod import Hexapod
import vrep
import cv2
import numpy as np
import sys
import time

from keycontrol import KeyControl


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


def key_loop():
    global STATE
    key = KeyControl()
    key.start()
    while True:
        state = key.get_key()
        if state is not None:
            STATE = state


def cap_loop(ver, rb, sleep_time):
    global lock, STATE
    key = KeyControl()
    cap = ImgCap(ver)
    key.start()
    while True:
        # lock.acquire()
        state = key.get_key()
        if state is not None:
            STATE = state
        try:
            cap.capture(rb.get_image(), STATE)
        except Exception as e:
            time.sleep(sleep_time)
            continue
        time.sleep(sleep_time)


def sample_data(ver, speed=0.05):
    client_id = connect(10)
    rb = Hexapod(client_id)
    vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)
    # time.sleep(1)
    rb.step_init()
    position_state = 0

    # creat thread
    global STATE, lock
    STATE = 0
    # lock=threading.Lock()
    cap_th = threading.Thread(
        target=cap_loop, args=(ver, rb, 0.05), name='cap_th')
    # key_th=threading.Thread(target=key_loop,name='key_th')

    cap_th.start()
    # key_th.start()

    while True:
        if STATE == 0:
            rb.one_step(0.003)
        #right
        if STATE == 1:
            rb.turn_left([5, 45])
        elif STATE == 2:
            rb.turn_left([20, 36])
        elif STATE == 3:
            rb.turn_left([20, 30])
        #left
        elif STATE == 4:
            rb.turn_right([5, 45])
        elif STATE == 5:
            rb.turn_right([20, 36])
        elif STATE == 6:
            rb.turn_right([20, 30])
        elif STATE==7:
            clean(ver)
        elif STATE==8:
            sys.exit(0)

def clean(ver):
    file_list = os.listdir('./image/{}'.format(self.ver))
    if not file_list == []:
        for name in file_list:
            os.remove('./image/{0}/{1}'.format(ver, name))

class ImgCap(object):
    def __init__(self, ver):
        self.index = 0
        self.ver = ver
        self.create_dir('./image/{}'.format(self.ver))

    def capture(self, data, label):
        data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(
            './image/{0}/image_{1}_{2}.jpg'.format(self.ver, self.index, label), data)
        self.index += 1

    def create_dir(self, dirs):
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        else:
            file_list = os.listdir('./image/{}'.format(self.ver))
            if not file_list == []:
                for name in file_list:
                    os.remove('./image/{0}/{1}'.format(self.ver, name))


def show_img(img):
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv2_test():
    img = cv2.imread('./image/0/image_4_0.jpg')


if __name__ == "__main__":
    sample_data(5, 0.01)
    # test_robot()
    # print(predict(model,'./image/0/image_16_1.jpg'))
    pass
