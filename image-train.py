import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
import cv2

import numpy as np
import os
import vrep
from hexapod import Hexapod, connect
import time
import threading


class ImageTrain(object):
    def build_model(self, X):
        model = keras.Sequential([
            keras.layers.Conv2D(64, (2, 2), activation='relu',
                                input_shape=X.shape[1:]),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, (2, 2), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(7, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy',
                    #   optimizer='adam',
                        optimizer=keras.optimizers.SGD(lr=0.00005),
                      metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])
        return model

    def lenet_5(self, input_shape=(32, 32, 1), classes=7):
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
                    #   optimizer='adam',
                        optimizer=keras.optimizers.SGD(lr=0.0003),
                      metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])
        return model

    def train(self,statr,end):
        # get data
        # label,data=self.get_data_range(statr,end)
        label,data=self.get_data('./gen_image/')

        permutation = np.random.permutation(label.shape[0])
        data = data[permutation]
        label = label[permutation]
        label,data=self.rechoice(label,data)
        # build model
        model = self.build_model(data)
        # model = self.lenet_5()
        # train
        # tbCallBack = keras.callbacks.TensorBoard(log_dir='.\logs')
        model.fit(data, label, batch_size=32, epochs=20,
                  validation_split=0.3)
        # save model
        model.save('result.h5')

    def get_data_range(self,start,end):
        data_list = []
        label_list = []
        base_name = './image/b/{}/'
        for i in range(start,end):
            name = base_name.format(i)
            label, data = self.get_data(name)
            data_list.append(data)
            label_list.append(label)
        data = np.concatenate(data_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        return label,data

    def get_data_slow(self, img_path='./image/1/'):
        # get file name
        img_size = (32, 32)
        label = []
        file_name_list = os.listdir(img_path)
        data = np.zeros((1,)+img_size+(1,), dtype=np.float32)
        i=0
        for file_name in file_name_list:
            # data
            file_name = img_path+file_name
            img = cv2.imread(file_name)
            img_resize = cv2.resize(img, img_size)
            img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

            # filter
            img_gray = cv2.medianBlur(img_gray, 3)

            img_array = np.array(img_gray)
            img_array = img_array.reshape((1,)+img_size+(1,))
            data = np.concatenate([data, img_array], axis=0)
            # label
            label_name = int(os.path.splitext(file_name)[0].split('_')[-1])
            if label_name == 7 or label_name == 8:
                label_name = 0
            # if label_name == 2:
            #     label_name = 3
            # if label_name == 5:
            #     label_name = 6
            label.append(label_name)
            i+=1
            print(i)

        data = np.delete(data, 0, axis=0)
        label = np.array(label)
        label = utils.to_categorical(label, num_classes=7)
        return label, data
        
    
    def get_data(self,img_path):
        img_size = (64, 64)
        file_name_list = os.listdir(img_path)
        data=np.zeros((len(file_name_list),)+img_size+(1,))
        label=np.zeros(len(file_name_list))
        i=0
        for file_name in file_name_list:
            # data
            file_name = img_path+file_name
            img = cv2.imread(file_name)
            img_resize = cv2.resize(img, img_size)
            img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

            # filter
            img_gray = cv2.medianBlur(img_gray, 3)

            img_array = np.array(img_gray)
            img_array = img_array.reshape((1,)+img_size+(1,))
            data[i]=img_array
            # label
            label_name = int(os.path.splitext(file_name)[0].split('_')[-1])
            if label_name == 7 or label_name == 8:
                label_name = 0
            # if label_name == 2:
            #     label_name = 3
            # if label_name == 5:
            #     label_name = 6
            label[i]=label_name
            i+=1
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

    def rechoice(self,label,data=None,num=6000):
        label=np.argmax(label,axis=1)
        e=np.random.choice(np.where(label==0)[0],5000)
        a=np.random.choice(np.where(label==2)[0],1900)
        b=np.random.choice(np.where(label==3)[0],1900)
        c=np.random.choice(np.where(label==5)[0],1900)
        d=np.random.choice(np.where(label==6)[0],1900)
        # f=np.random.choice(np.where(label==1)[0],300)
        # g=np.random.choice(np.where(label==4)[0],200)
        result=np.concatenate([a,b,c,d,e])
        # result=np.concatenate([c,d])
        np.random.shuffle(result)
        # result=np.concatenate([b,d])
        # result=np.concatenate([e,f,g])
        label=label[result]
        label=utils.to_categorical(label, num_classes=7)
        if data is  None:
            return label
        data=data[result]
        return label,data



def test_robot():
    client_id = connect(10)
    rb = Hexapod(client_id)

    vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)
    time.sleep(1)
    rb.step_init()

    # creat thread
    global state, lock
    state = 0
    lock = threading.Lock()
    cap_th = threading.Thread(target=refresh_state, args=(rb,))
    cap_th.start()

    while True:
        lock.acquire()
        try:
            if state == 0:
                rb.one_step(0.002)
            # left
            if state == 1:
                rb.turn_left([10, 40])
            elif state == 2:
                # rb.turn_left([20, 34])
                rb.turn_left([20, 36])
            elif state == 3:
                rb.turn_left([20, 29])
                # rb.turn_left([20, 25])
            # right
            elif state == 4:
                rb.turn_right([10, 40])
            elif state == 5:
                # rb.turn_right([20, 34])
                rb.turn_right([20, 36])
            elif state == 6:
                rb.turn_right([20, 29])
                # rb.turn_right([20, 25])
        finally:
            lock.release()
            pass
        print(state)

def control(img):
    # step one 
    from utils import cv2_util
    img_binary=cv2_util.binarization(img)
    left,right=cv2_util.get_px_num(img_binary)
    delta=left-right
    print(delta)
    if delta>300:
        result=3
    elif delta<-300:
        result=6
    elif delta>200:
        result=2
    elif delta<-200:
        result=5
    else:
        result=0
    return result

def refresh_state(rb):
    global state, lock
    model = keras.models.load_model('result.h5')
    cv2.namedWindow('sensor')
    while True:
        img = rb.get_image()
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_resize = cv2.resize(img_gray, (64, 64))
        # filter
        img_gray = cv2.medianBlur(img_resize, 3)
        img_gray=np.rot90(img_gray)
        img_gray=np.rot90(img_gray)

        # from utils.cv2_util import edge_det
        # img_gray=edge_det(img_gray)

        # from utils.cv2_util import img_show
        # img_show(img_gray)
        result=0
        if type(img_gray) is np.ndarray:
            cv2.imshow('sensor',cv2.resize(img_gray.astype(np.uint8),(320,320)))
            cv2.waitKey(1)

            float_data = img_resize.reshape(1, 64, 64, 1)
            data=float_data.astype(np.uint8)
            result = np.argmax(model.predict(data))

        # result=control(img_gray)
        
        lock.acquire()

        try:
            state = result
        finally:
            lock.release()


def train_set_percentage():
    test=ImageTrain()
    label,data=test.get_data_fast('./gen_image/')
    label=np.argmax(label,axis=1)
    a=np.size(np.where(label==2))
    b=np.size(np.where(label==3))
    c=np.size(np.where(label==5))
    d=np.size(np.where(label==6))
    e=np.size(np.where(label==0))
    f=np.size(np.where(label==1))
    g=np.size(np.where(label==4))
    total=a+b+c+d+e
    print('2:{} 3:{} 5:{} 6:{} 0:{} 1:{} 4:{}'.format(a,b,c,d,e,f,g))



if __name__ == "__main__":
    # 2:6167 3:7705 5:7340 6:6739 0:15211
    # file_range=(0,19)
    # train_set_percentage()
    # train = ImageTrain()
    # train.train(0,31)
    # train.rrr()
    # for i in range(31):
    # train.evaluate(17)
    # train_set_percentage()
    test_robot() 
    # train_set_percentage()
    pass
