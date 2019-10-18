import cv2
from bluetooth_control import Bluez
from tensorflow import keras
import numpy as np


class RobotControl(object):
    def __init__(self):
        self.state = 0
        self.cap_url = 'http://192.168.1.1:4455/?action=stream'

        self.cap=cv2.VideoCapture(self.cap_url)
        # self.cap = cv2.VideoCapture(1)
        self.bz = Bluez()
        # load model
        self.line_tracker_model = keras.models.load_model('./result_v1.1.h5')
        self.sa_det_mode = keras.models.load_model(
            './turn_straight_cla_v0.2.h5')

    def bluetooth_connect(self):
        while not self.bz.is_bluetooth_connected:
            try:
                self.bz.connect(5)
            except ConnectionError:
                pass
        print('connect success')

    def get_state(self, show_window=False):
        assert(self.cap is not None)
        ret, img = self.cap.read()
        if ret:
            img_resize = cv2.resize(img, (64, 64))
            img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

            # filter
            img_gray = cv2.medianBlur(img_gray, 3)

            # binary
            img_gray = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)[1]

            #reverse the color
            img_gray = 255 - img_gray

            # predict
            float_data = img_gray.reshape(1, 64, 64, 1)
            data = float_data.astype(np.uint8)
            if np.argmax(self.sa_det_mode.predict(data)) == 1:
                state = np.argmax(self.line_tracker_model.predict(data))
            else:
                state = 0
            self.state = state

            if show_window:
                # img_rgb=cv2.cvtColor(img_resize,cv2.COLOR_GRAY2RGB)
                img_rgb=img_resize
                img_num = cv2.putText(cv2.resize(
                    cv2.threshold(img_rgb, 30, 255, cv2.THRESH_BINARY)[1].astype(np.uint8),(320, 320)), str(self.state), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('sensor',img_num )
                cv2.waitKey(1)

    def loop(self):
        while True:
            try:
                self.get_state(show_window=True)
            except:
                self.cap=cv2.VideoCapture(self.cap_url)
            print(self.state, end='\r')
            try:
                self.bz.send(self.state)
            except:
                self.bluetooth_connect()
                


if __name__ == "__main__":
    control = RobotControl()
    control.loop()
