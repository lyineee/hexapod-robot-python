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

if __name__ == "__main__":
    # label = np.load('./data/label_middle.npy')
    # data = np.load('./data/data_middle.npy')
    # model=build_model()
    # label = label.reshape(label.shape[0], 1)
    # history = model.fit(
    #     label, data, epochs=150, batch_size=5, validation_split=0.05)
    # model.save('./data/NN_middle.h5')

    graph('back')
    #TODO maximum: generate_use_data([[-100,15],[-40,40],[-10,100]],50)
    # generate_use_data([[-30,5],[-25,25],[0,30]],50)
    pass

