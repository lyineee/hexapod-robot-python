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
        self.line_tracker_model = keras.models.load_model('./result_v1.0.h5')
        self.sa_det_mode = keras.models.load_model(
            './turn_straight_cla_v0.2.h5')
        self.first_flag = True

    def bluetooth_connect(self):
        self.bz.connect(5)

    def get_state(self, show_image_window=None):
        assert(self.cap is not None)
        # TODO ret judge
        ret, img = self.cap.read()
        if ret:
            img_resize = cv2.resize(img, (64, 64))
            img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

            # filter
            img_gray = cv2.medianBlur(img_gray, 3)

            # bin
            img_gray = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)[1]
            img_gray = 255 - img_gray

            # img_gray = img_gray+128
            # img_gray = np.rot90(img_gray)
            # img_gray = np.rot90(img_gray)
            # img_gray = cv2.flip(img_gray, 1)

            # predict
            # cv2.imshow('origin', cv2.resize(
            #     img.astype(np.uint8), (320, 320)))
            float_data = img_gray.reshape(1, 64, 64, 1)
            data = float_data.astype(np.uint8)
            if np.argmax(self.sa_det_mode.predict(data)) == 1:
                state = np.argmax(self.line_tracker_model.predict(data))
            else:
                state = 0
            self.state = state

            if show_image_window is not None:
                img_num = cv2.putText(cv2.resize(
                    cv2.threshold(img_resize, 128, 255, cv2.THRESH_BINARY)[1].astype(np.uint8),(320, 320)), str(self.state), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('sensor',img_num )
                cv2.waitKey(1)

    def loop(self):
        while True:
            self.get_state(show_image_window=True)
            print(self.state, end='\r')
            self.bz.send(self.state)


if __name__ == "__main__":
    control = RobotControl()
    # control.cap=
    control.bluetooth_connect()
    control.loop()
