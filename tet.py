from bluetooth_control import Bluez
import numpy as np 
import cv2 
import time
class Test(object):
    cache=np.zeros((16,),dtype=np.int16)
    test=0
    data_args=15
    def __init__(self):
        self.blue=Bluez()
        self.blue.connect(10)
        cmd=self.test

    def start(self):
        cv2.namedWindow('test',flags=cv2.WINDOW_NORMAL)
        time.sleep(1)
        cv2.createTrackbar('args','test',20,500,self.track_bar_callback)
        cv2.createTrackbar('cmd','test',0,15,self.cmd_callback)


    def send_data(self):
        send_1=str(self.test)
        send_2=str(self.data_args)
        send_data='|'+send_1.zfill(2)+send_2.zfill(3)
        # print(send_data)
        self.blue.send(send_data)
    def track_bar_callback(self,data):
        self.cache[self.test]=data
        self.data_args=round(data/10)
        self.send_data()

    def cmd_callback(self,data):
        self.test=data
        cv2.setTrackbarPos('args','test',self.cache[data])
        self.data_args=self.cache[data]
        self.send_data()

    def loop(self):
        self.start()
        try:
            while True:
                cv2.waitKey(1)
                time.sleep(0.2)
                self.send_data()
        finally:
            cv2.destroyAllWindows()

data=7
def cback(data_t):
    global data
    data=data_t
if __name__ == "__main__":
    # test=Test()
    # test.loop()
    blue=Bluez()
    blue.connect(5)
    cv2.namedWindow('test',flags=cv2.WINDOW_NORMAL)
    cv2.createTrackbar('cmd','test',7,15,cback)
    while True:
        blue.send(str(data))
        print(data)
        cv2.waitKey(1)
        time.sleep(0.3)