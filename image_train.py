# %%
import os
import threading
import time
from queue import Queue

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils

import vrep
from hexapod import Hexapod, connect

# %%


class ImageTrain(object):

    def build_model(self, data, label, lr=0.0005):
        model = keras.Sequential([
            keras.layers.Conv2D(64, (2, 2), activation='relu',
                                input_shape=data.shape[1:]),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(label.shape[-1], activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy',
                      # optimizer='adam',
                      #   optimizer=keras.optimizers.SGD(lr=0.00005),
                      optimizer=keras.optimizers.SGD(lr),
                      metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])
        return model

    def lenet_5(self, input_shape=(64, 64, 1), classes=7):
        X_input = keras.layers.Input(input_shape)
        X = keras.layers.ZeroPadding2D((1, 1))(X_input)
        X = keras.layers.Conv2D(6, (5, 5), strides=(
            1, 1), padding='valid', name='conv1')(X)
        X = keras.layers.Activation('tanh')(X)
        X = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(X)
        X = keras.layers.Conv2D(6, (5, 5), strides=(
            1, 1), padding='valid', name='conv2')(X)
        X = keras.layers.Activation('tanh')(X)
        X = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(X)
        X = keras.layers.Flatten()(X)
        X = keras.layers.Dense(120, activation='tanh', name='fc1')(X)
        X = keras.layers.Dense(84, activation='tanh', name='fc2')(X)
        X = keras.layers.Dense(classes, activation='softmax')(X)
        model = keras.Model(inputs=X_input, outputs=X, name='lenet_5')
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      # optimizer=keras.optimizers.SGD(lr=0.0003),
                      metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])
        return model

    def train(self, data=None, label=None):
        # get data
        # label,data=self.get_data_range(statr,end)
        if data is None or label is None:
            label, data = self.get_data('./gen_image/')

        permutation = np.random.permutation(label.shape[0])
        data = data[permutation]
        label = label[permutation]
        label, data = self.rechoice(
            label, data, num=[5000, 0, 2300, 2300, 0, 2300, 2300])
        # build model
        model = self.build_model(data, label)
        # model = self.lenet_5()
        # train
        # tbCallBack = keras.callbacks.TensorBoard(log_dir='.\logs')
        model.fit(data, label, batch_size=32, epochs=20,
                  validation_split=0.3)
        # save model
        model.save('result.h5')

    def get_data_range(self, dir_name, start, end):
        data_list = []
        label_list = []
        base_name = dir_name+'{}/'
        for i in range(start, end):
            name = base_name.format(i)
            label, data = self.get_data(name)
            data_list.append(data)
            label_list.append(label)
        data = np.concatenate(data_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        return label, data


    def get_data(self, img_path):
        img_size = (64, 64)
        file_name_list = os.listdir(img_path)
        data = np.zeros((len(file_name_list),)+img_size+(1,))
        label = np.zeros(len(file_name_list))
        i = 0
        L = 80
        N = len(file_name_list)
        for file_name in file_name_list:
            # data
            file_name = img_path+file_name
            img = cv2.imread(file_name)
            img_resize = cv2.resize(img, img_size)
            img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

            # filter
            img_gray = cv2.medianBlur(img_gray, 3)

            # TODO binary
            img_gray = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)[1]

            img_array = np.array(img_gray)
            img_array = img_array.reshape((1,)+img_size+(1,))
            data[i] = img_array
            # label
            label_name = int(os.path.splitext(file_name)[0].split('_')[-1])
            if label_name == 7 or label_name == 8:
                label_name = 0
            # if label_name == 2:
            #     label_name = 3
            # if label_name == 5:
            #     label_name = 6
            label[i] = label_name
            i += 1
            print("{{{0}>{1}}} {2}% ".format('='*round(i*L/N),
                                             '.'*round((N-i)*L/N), round(i*100/N)), end="\r")
        label = utils.to_categorical(label, num_classes=7)
        return label, data

    def predict(self, model, filename):
        img = cv2.imread(filename)
        img_resize = cv2.resize(img, (64, 64))
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_array = np.array(img_gray)
        # for i in range(img_array.shape[0]):
        #     for j in range(img_array.shape[1]):
        #         img_array[i][j]=255-img_array[i][j]
        result = img_array.reshape(-1, 64, 64, 1)
        return model.predict(result)

    def evaluate(self, var=0):
        label, data = self.get_data('./image/b/{}/'.format(var))
        # label, data = self.get_data_range(266,274)
        # label,data=self.rechoice(label,data,40)
        model = keras.models.load_model('result.h5')
        # ues evaluate
        # model.evaluate(data,label)

        # use predict
        result = model.predict(data)
        total = len(label)
        hit = 0
        for i in range(total):
            if np.argmax(label[i]) == np.argmax(result[i]):
                hit += 1
        print('total: {} hit: {} ,accuracy:{}'.format(total, hit, hit/total))

    def rechoice(self, label, data=None, num=None, shuffle=True, num_classes=7):
        assert(num != None)
        label = np.argmax(label, axis=1)
        con_list = []
        data_list = [0, 1, 2, 3, 4, 5, 6]
        for i, element in enumerate(num):
            con_list.append(np.random.choice(
                np.where(label == data_list[i])[0], element))
        # e = np.random.choice(np.where(label == 0)[0],  num[0])
        # a = np.random.choice(np.where(label == 2)[0],  num[1])
        # b = np.random.choice(np.where(label == 3)[0],  num[2])
        # c = np.random.choice(np.where(label == 5)[0],  num[3])
        # d = np.random.choice(np.where(label == 6)[0],  num[4])
        # result = np.concatenate([a, b, c, d, e])
        result = np.concatenate(con_list)
        if shuffle:
            np.random.shuffle(result)
        label = label[result]
        label = utils.to_categorical(label, num_classes=num_classes)
        if data is None:
            return label
        data = data[result]
        return label, data

#global state
state=0
def test_robot(model_name='result.h5',rb=None):
    # global START_FLAG
    if rb is None:
        client_id = connect(10)
        rb = Hexapod(client_id)
    # model = keras.models.load_model('result.h5')
    rb.start_simulation()
    time.sleep(1)
    # rb.step_init()

    # creat thread
    global state, lock
    lock = threading.Lock()
    cap_th = threading.Thread(target=refresh_state, args=(rb, model_name))
    cap_th.start()
    stage = 1

    rb.delay=0.0087107652327172
    rb.step_init()
    while True:
        time_s = time.time()
        lock.acquire()
        try:
            state_t = state
        finally:
            lock.release()

        if state_t == 0:
            # rb.one_step_t(0.013, stage)
            rb.ahead()
        # left
        if state_t == 1:
            rb.turn_left([10, 28], stage,time_0=0.013)
        elif state_t == 2:
            # rb.turn_left([20, 39],stage)
            # rb.turn_left([10, 23], stage,time_0=0.013)
            rb.left_1()

        elif state_t == 3:
            # rb.turn_left([20, 25],stage)
            # rb.turn_left([10, 16], stage,time_0=0.013)
            # rb.turn_left([10, 13], stage,time_0=0.013)
            rb.left_2()

        # right
        elif state_t == 4:
            rb.turn_right([10, 28], stage,time_0=0.013)
        elif state_t == 5:
            # rb.turn_right([20, 39],stage)
            # rb.turn_right([10, 23], stage,time_0=0.013)
            rb.right_1()
        elif state_t == 6:
            # rb.turn_right([20, 25],stage)
            # rb.turn_right([10, 16], stage,time_0=0.013)
            # rb.turn_right([10, 13], stage,time_0=0.013)
            rb.right_2()

        # rb.show_speed()
        print('state:{}'.format(state), 'step_time_use:{}'.format(
            str(time.time()-time_s)), end='\r')
        # print('step_time_use:{}'.format(str(time.time()-)),end='\r')

        
lock=threading.Lock()

def control(img):
    # step one
    from utils import cv2_util
    img_binary = cv2_util.binarization(img)
    left, right = cv2_util.get_px_num(img_binary)
    delta = left-right
    print(delta)
    if delta > 300:
        result = 3
    elif delta < -300:
        result = 6
    elif delta > 200:
        result = 2
    elif delta < -200:
        result = 5
    else:
        result = 0
    return result


def refresh_state(cap, model_name,q=None):
    global state, lock
    model_1 = keras.models.load_model(model_name)
    model = keras.models.load_model('./result_v1.1.h5')
    cv2.namedWindow('sensor')
    if q is not None:
        q.put('init_complete')
    while True:
        img = cap.get_image()

        # lock.acquire()
        # try:
        #     img = cap.get_image()
        # except:
        #     continue
        # finally:
        #     lock.release()
        # ret,img = cap.read()

        img_resize = cv2.resize(img, (64, 64))
        # img_resize=np.rot90(img_resize)
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

        # filter
        img_gray = cv2.medianBlur(img_gray, 3)
        img_gray = np.rot90(img_gray)
        # img_gray=np.rot90(img_gray)
        # img_gray=np.rot90(img_gray)
        img_gray = cv2.flip(img_gray, 1)

        img_gray = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)[1]

        # from utils.cv2_util import edge_det
        # img_gray=edge_det(img_gray)

        # from utils.cv2_util import img_show
        # img_show(img_gray)
        result = 0
        float_data = img_gray.reshape(1, 64, 64, 1)
        data = float_data.astype(np.uint8)
        if np.argmax(model_1.predict(data)) == 1:
            result = np.argmax(model.predict(data))
        else:
            result = 0
            # print(result)

        # result=control(img_gray)

        lock.acquire()
        try:
            state = result
        finally:
            lock.release()

        img_rgb=cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
        img_text=cv2.putText(cv2.resize(img_rgb.reshape(64, 64,3).astype(np.uint8), (320, 320)), str(
            state), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('sensor', img_text)
        cv2.waitKey(1)
        time.sleep(0.2)

def train_set_percentage(label=None):
    test = ImageTrain()
    if label is None:
        label, data = test.get_data('./gen_image/')
    label = np.argmax(label, axis=1)
    a = np.size(np.where(label == 2))
    b = np.size(np.where(label == 3))
    c = np.size(np.where(label == 5))
    d = np.size(np.where(label == 6))
    e = np.size(np.where(label == 0))
    f = np.size(np.where(label == 1))
    g = np.size(np.where(label == 4))
    # total = a+b+c+d+e
    print('2:{} 3:{} 5:{} 6:{} 0:{} 1:{} 4:{}'.format(a, b, c, d, e, f, g))


# %%
if __name__ == "__main__":
    # 2:6167 3:7705 5:7340 6:6739 0:15211
    # train_set_percentage(label)
    # train_set_percentage()
    # %%
    # train = ImageTrain()
    # %%
    # label, data = train.get_data('./image/archive-normal/')
    # label,data=train.get_data_range('./image/archive-normal/{}/',0,176)
    # %%
    # train.train(data=data,label=label)
    # train.train()
    # %%
    test_robot('turn_straight_cla_v0.2.h5')
    # refresh_state(cv2.VideoCapture('http://192.168.1.1:4455/?action=stream'),'turn_straight_cla_v0.2.h5')
    pass