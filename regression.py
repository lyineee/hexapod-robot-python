import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(1,)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(2)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.0005)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

def graph(name):

    label=np.load('./data/label_{}.npy'.format(name))
    data=np.load('./data/data_{}.npy'.format(name))

    model=keras.models.load_model('./data/NN_{}.h5'.format(name))

    predict_data=model.predict(label)

    data=np.concatenate([data,predict_data],axis=1)

    plt.plot(label,data)
    plt.show()

def generate_use_data(sample_range,sample_rate):
    pre=['./data/NN_ahead','./data/NN_middle','./data/NN_back']
    for i in range(3):
        model=keras.models.load_model('{}.h5'.format(pre[i]))
        sampling_point=np.linspace(min(sample_range[i]),max(sample_range[i]),sample_rate)
        sampling_point=sampling_point.reshape(sampling_point.shape[0],1)
        predict=model.predict(sampling_point)
        sampling_point= np.around(sampling_point,decimals=1)
        predict= np.around(predict,decimals=1)
        result=np.concatenate([sampling_point,predict],axis=1)
        np.save('{}_use.npy'.format(pre[i]),result)

def gen_data_for_arduino():
    pre=['./data/NN_ahead','./data/NN_middle','./data/NN_back']
    for i in range(3):
        row_data=np.load('{}_use.npy'.format(pre[i]))
        #because of the different start angle of the leg
        # row_data=row_data+[0,0,45]
        # change the second data
        row_data[:,1]=-row_data[:,1]
        if pre[i]=='./data/NN_ahead':
            row_data[:,0]=row_data[:,0]-90
        # convert into arduino data
        row_result=512/225*row_data+307.2
        row_result=row_result.astype(np.int32)
        result=str(row_result).replace('[','{').replace(']','}').replace(' ',',')
        with open('{}_arduino.txt'.format(pre[i]),'w') as f:
            f.write(result)

        


if __name__ == "__main__":
    # label = np.load('./data/label_ahead.npy')
    # data = np.load('./data/data_ahead.npy')
    # model=build_model()
    # label = label.reshape(label.shape[0], 1)
    # history = model.fit(
    #     label, data, epochs=150, batch_size=5, validation_split=0.05)
    # model.save('./data/NN_ahead.h5')

    # label = np.load('./data/label_middle.npy')
    # data = np.load('./data/data_middle.npy')
    # model=build_model()
    # label = label.reshape(label.shape[0], 1)
    # history = model.fit(
    #     label, data, epochs=150, batch_size=5, validation_split=0.05)
    # model.save('./data/NN_middle.h5')

    # label = np.load('./data/label_back.npy')
    # data = np.load('./data/data_back.npy')
    # model=build_model()
    # label = label.reshape(label.shape[0], 1)
    # history = model.fit(
    #     label, data, epochs=250, batch_size=7, validation_split=0.05)
    # model.save('./data/NN_back.h5')

    # graph('back')
    # graph('middle')
    # graph('ahead')
    # -10 - 100 55 - -55   
    generate_use_data([[-10,45],[-25,30],[-50,10]],50)
    # gen_data_for_arduino()
    pass

