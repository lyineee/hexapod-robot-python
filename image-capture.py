import os
import sys
import threading
import time

import cv2
import numpy as np
import pygame

# git pull --rebase origin master
import vrep
from hexapod import Hexapod
from keycontrol import KeyControl

first_flag = True


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


def only_control_loop(ver, rb, sleep_time):
    global lock, STATE
    key = KeyControl()
    key.start()
    while True:
        state = key.get_key()
        if state is not None:
            STATE = state
        time.sleep(0.2)


def cap_loop(ver, rb, sleep_time):
    global lock, STATE
    key = KeyControl()
    cap = ImgCap(ver)
    if ver == 0:
        key.first_run()
    key.start()
    while True:
        if STATE == -1:
            cap.create_dir()
            print('clean save dir')
            while key.get_key() != 10:
                pass
            # rb.start_simulation()
            STATE = 10
        if STATE == -2:
            print('thread exit')
            key.end()
            sys.exit(0)
        state = key.get_key()
        if state is not None:
            STATE = state
        lock.acquire()
        try:
            cap.capture(rb.get_image(), STATE)
        except Exception as e:
            time.sleep(sleep_time)
            continue
        finally:
            lock.release()
        time.sleep(sleep_time)


def sample_data(ver, rb, speed=0.05):

    rb.step_init()
    # creat thread
    global STATE, lock
    STATE = -1  # cap stop and clean the dir
    lock = threading.Lock()
    cap_th = threading.Thread(
        target=cap_loop, args=(ver, rb, 0.05), name='cap_th')

    # cap_th = threading.Thread(
    #     target=only_control_loop, args=(ver, rb, 0.05), name='cap_th')
    cap_th.start()

    while True:
        # if STATE == 0:
        #     rb.one_step(0.003)
        # #right
        # if STATE == 1:
        #     rb.turn_left([10, 40])
        # elif STATE == 2:
        #     rb.turn_left([20, 34])
        # elif STATE == 3:
        #     rb.turn_left([20, 30])
        # #left
        # elif STATE == 4:
        #     rb.turn_right([10, 40])
        # elif STATE == 5:
        #     rb.turn_right([20, 34])
        # elif STATE == 6:
        #     rb.turn_right([20, 30])
        if STATE == 0:
            rb.one_step(0.002)
        # left
        if STATE == 1:
            rb.turn_left([15, 40])
        elif STATE == 2:
            rb.turn_left([20, 34])
        elif STATE == 3:
            rb.turn_left([20, 30])
        # right
        elif STATE == 4:
            rb.turn_right([15, 40])
        elif STATE == 5:
            rb.turn_right([20, 34])
        elif STATE == 6:
            rb.turn_right([20, 30])
        elif STATE == 7:
            STATE = -1
            rb.stop_simulation()
        elif STATE == 8:
            STATE = -2
            cap_th.join()
            rb.stop_simulation()
            sys.exit(0)
        elif STATE == 9:
            # ver+=1
            # while STATE!=10:
            #     pass
            # rb.start_simulation()
            STATE = -2  # end thread
            cap_th.join()
            rb.stop_simulation()
            time.sleep(0.5)
            # cap_th = threading.Thread(
            #     target=cap_loop, args=(ver, rb, 0.05), name='cap_th')
            # cap_th.start()
            return
        elif STATE == 10:
            rb.start_simulation()
            pass


class ImgCap(object):
    def __init__(self, ver):
        self.index = 0
        self.ver = ver
        self.base_dir = './image/c/'
        self.save_dir = self.base_dir+str(self.ver)
        self.create_dir()

    def capture(self, data, label):
        # data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(
            '{}/image_{}_{}.jpg'.format(self.save_dir, self.index, label), data)
        self.index += 1

    def create_dir(self, dirs=None):
        if dirs == None:
            dirs = self.save_dir
        self.index = 0
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        else:
            file_list = os.listdir(dirs)
            if not file_list == []:
                for name in file_list:
                    os.remove('{0}/{1}'.format(self.save_dir, name))


def show_img(img):
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv2_test():
    img = cv2.imread('./image/0/image_4_0.jpg')


if __name__ == "__main__":
    ver = 0
    client_id = connect(10)
    rb = Hexapod(client_id)
    # pygame.init()
    # time.sleep(1)
    # vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)
    # time.sleep(1)
    while True:
        sample_data(ver, rb, 0.01)
        ver += 1
    # test_robot()
    # print(predict(model,'./image/0/image_16_1.jpg'))
    pass
