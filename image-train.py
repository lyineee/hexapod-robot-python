import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
import cv2 



class ImageTrain(object):
    def build_model(self,X):
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
            keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                    #   optimizer=keras.optimizers.SGD(learning_rate=0.0007),
                      metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])
        return model

    def lenet_5(self,input_shape=(32,32,1),classes=3):
        X_input=keras.layers.Input(input_shape)
        X=keras.layers.ZeroPadding2D((1,1))(X_input)
        X=keras.layers.Conv2D(6,(5,5),strides=(1,1),padding='valid',name='conv1')(X)
        X=keras.layers.Activation('tanh')(X)
        X=keras.layers.MaxPooling2D((2,2),strides=(2,2))(X)
        X=keras.layers.Conv2D(6,(5,5),strides=(1,1),padding='valid',name='conv2')(X)
        X=keras.layers.Activation('tanh')(X)
        X=keras.layers.MaxPooling2D((2,2),strides=(2,2))(X)
        X=keras.layers.Flatten()(X)
        X=keras.layers.Dense(120,activation='tanh',name='fc1')(X)
        X=keras.layers.Dense(84,activation='tanh',name='fc2')(X)
        X=keras.layers.Dense(classes,activation='softmax')(X)
        model=keras.Model(inputs=X_input,outputs=X,name='lenet_5')
        model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                #   optimizer=keras.optimizers.SGD(learning_rate=0.0007),
                  metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])
        return model

    def train(self):
        # get data
        data_list=[]
        label_list=[]
        base_name='./image/{}/'
        for i in range(4):
            name=base_name.format(i)
            label,data=get_data(name)
            data_list.append(data)
            label_list.append(label)
        data=np.concatenate(data_list,axis=0)
        label=np.concatenate(label_list,axis=0)
        # build model
        # model = build_model(data)
        model = lenet_5()
        # train
        tbCallBack = keras.callbacks.TensorBoard(log_dir='.\logs')
        model.fit(data, label, batch_size=32, epochs=40,
                  validation_split=0.1, callbacks=[tbCallBack])
        # save model
        model.save('result.h5')


    def get_data(self,img_path='./image/1/'):
        # get file name
        img_size = (32, 32)
        label = []
        file_name_list = os.listdir(img_path)
        data = np.zeros((1,)+img_size+(1,), dtype=np.float32)
        for file_name in file_name_list:
            # data
            file_name = img_path+file_name
            img = cv2.imread(file_name)
            img_resize = cv2.resize(img, img_size)
            img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

            # filter
            img_gray=cv2.medianBlur(img_gray,3)

            img_array = np.array(img_gray)
            img_array = img_array.reshape((1,)+img_size+(1,))
            data = np.concatenate([data, img_array], axis=0)
            # label
            label_name = int(os.path.splitext(file_name)[0].split('_')[-1])
            label.append(label_name)

        data = np.delete(data, 0, axis=0)
        label = np.array(label)
        label=utils.to_categorical(label,num_classes=3)
        return label, data

    def predict(self,model,filename):
        img=cv2.imread(filename)
        img_resize=cv2.resize(img,(64,64))
        img_gray = cv2.cvtColor(img_resize,cv2.COLOR_RGB2GRAY)
        img_array=np.array(img_gray)
        # for i in range(img_array.shape[0]):
        #     for j in range(img_array.shape[1]):
        #         img_array[i][j]=255-img_array[i][j]
        result=img_array.reshape(-1,64,64,1)
        return model.predict(result)




    def test(self):
        label,data=get_data('./image/4/')
        model=keras.models.load_model('result.h5')
        result=model.predict(data)
        total=len(label)
        hit=0
        for i in range(total):
            if np.argmax(label[i])==np.argmax(result[i]):
                hit+=1
        print('total: {} hit: {} ,accuracy:{}'.format(total,hit,hit/total))

if __name__ == "__main__":
    pass