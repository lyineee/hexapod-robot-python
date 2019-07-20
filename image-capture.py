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
    key=KeyControl()
    key.start()
    while True:
        state=key.get_key()
        if state is not None:
            STATE=state

            
def cap_loop(ver,rb,sleep_time):
    global lock,STATE
    key=KeyControl()
    cap=ImgCap(ver)
    key.start()
    while True:
        # lock.acquire()
        state=key.get_key()
        if state is not None:
            STATE=state
        try:
            cap.capture(rb.get_image(),STATE)
        except Exception as e:
            time.sleep(sleep_time)
            continue
        time.sleep(sleep_time)


def sample_data(ver,speed=0.05):
    client_id = connect(10)
    rb = Hexapod(client_id)
    vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)
    # time.sleep(1)
    rb.step_init()
    position_state = 0

    #creat thread
    global STATE,lock
    STATE=0
    # lock=threading.Lock()
    cap_th=threading.Thread(target=cap_loop,args=(ver,rb,0.05),name='cap_th')
    # key_th=threading.Thread(target=key_loop,name='key_th')

    cap_th.start()
    # key_th.start()

    while True:
        if STATE==0:
            rb.one_step(0.003)
        if STATE==1:
            rb.turn_left([20, 30])
        if STATE==2:
            rb.turn_right([20,36])


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





def test_robot():
    client_id = connect(10)
    rb = Hexapod(client_id)

    vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)
    time.sleep(1)
    rb.step_init()

    #creat thread
    global state,lock
    state=0 
    lock=threading.Lock()
    cap_th=threading.Thread(target=refresh_state,args=(rb,))
    cap_th.start()

    while True:
        lock.acquire()
        try:
            if state==0:
                rb.one_step(0.01)
            elif state==1:
                rb.go_left()
            elif state==2:
                rb.go_right()
        finally:
            lock.release()
        print(state)

def refresh_state(rb):
    global state,lock
    model=keras.models.load_model('result.h5')
    while True:
        img=rb.get_image()
        img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img_resize=cv2.resize(img_gray,(32,32))
        data=img_resize.reshape(1,32,32,1)
        result=np.argmax(model.predict(data))
        lock.acquire()
        try:
            state=result
        finally:
            lock.release()

def show_img(img):
    cv2.namedWindow("Image") 
    cv2.imshow("Image", img) 
    cv2.waitKey (0)
    cv2.destroyAllWindows()

def cv2_test():
    img=cv2.imread('./image/0/image_4_0.jpg')


        

if __name__ == "__main__":
    sample_data(5,0.01)
    # test_robot()
    # print(predict(model,'./image/0/image_16_1.jpg'))
    pass