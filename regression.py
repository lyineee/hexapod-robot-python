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
    # optimizer = tf.keras.optimizers.RMSprop(0.0005)

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  # optimizer=keras.optimizers.SGD(lr=0.00005),
                  metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])
    return model


def model_2():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(1,)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(3)
    ])
    # optimizer = tf.keras.optimizers.RMSprop(0.0005)

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  # optimizer=keras.optimizers.SGD(lr=0.00005),
                  metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])
    return model


def graph(name):

    label = np.load('./data/label_{}.npy'.format(name))
    data = np.load('./data/data_{}.npy'.format(name))

    model = keras.models.load_model('./data/NN_{}.h5'.format(name))

    predict_data = model.predict(label)

    data = np.concatenate([data, predict_data], axis=1)

    plt.plot(label, data)
    plt.show()


def generate_use_data(sample_range, sample_rate):
    pre = ['./data/NN_ahead', './data/NN_middle', './data/NN_back']
    for i in range(3):
        model = keras.models.load_model('{}.h5'.format(pre[i]))
        sampling_point = np.linspace(
            min(sample_range[i]), max(sample_range[i]), sample_rate)
        sampling_point = sampling_point.reshape(sampling_point.shape[0], 1)
        predict = model.predict(sampling_point)
        sampling_point = np.around(sampling_point, decimals=1)
        predict = np.around(predict, decimals=1)
        result = np.concatenate([sampling_point, predict], axis=1)
        np.save('{}_use.npy'.format(pre[i]), result)


def generate_use_data_t(sample_range, sample_rate, model_all=None):
    pre = ['./data/NN_ahead', './data/NN_middle', './data/NN_back']
    for i in range(3):
        if model_all is None:
            model = keras.models.load_model('{}.h5'.format(pre[i]))
        else:
            model = model_all[i]
        sampling_point = np.linspace(
            sample_range[i][0], sample_range[i][1], sample_rate)
        sampling_point = sampling_point.reshape(sampling_point.shape[0], 1)
        predict = model.predict(sampling_point)
        predict = np.around(predict, decimals=1)
        np.save('{}_use.npy'.format(pre[i]), predict)


def gen_data_for_arduino():
    data_range = [80, 470]
    pre = ['./data/NN_ahead', './data/NN_middle', './data/NN_back']
    for i in range(3):
        row_data = np.load('{}_use.npy'.format(pre[i]))
        # because of the different start angle of the leg
        # row_data=row_data+[0,0,45]
        # change the second data
        row_data[:, 1] = -row_data[:, 1]
        if pre[i] == './data/NN_ahead':
            row_data[:, 0] = row_data[:, 0]-45
        # convert into arduino data
        # row_result=512/225*row_data+307.2
        # row_result=map(row_data,[-90,90],data_range)
        row_result = row_result.astype(np.int32)
        result = str(row_result).replace(
            '[', '{').replace(']', '}').replace(' ', ',')
        with open('{}_arduino.txt'.format(pre[i]), 'w') as f:
            f.write(result)


def gen_test():
    SERVOMAX_R = 510
    SERVOMIN_R = 92

    SERVOMAX_L = 467
    SERVOMIN_L = 92
    # reverse the ahead and back
    aheadRow = np.load('./data/NN_back_use.npy')
    middleRow = np.load('./data/NN_middle_use.npy')
    backRow = np.load('./data/NN_ahead_use.npy')


    aheadRow[:, 1] = -aheadRow[:, 1]
    middleRow[:, 1] = -middleRow[:, 1]
    backRow[:, 1] = -backRow[:, 1]

    # aheadRow[:, 0] = aheadRow[:, 0][::-1]
    # middleRow[:, 0] = middleRow[:, 0][::-1]
    # backRow[:, 0] = backRow[:, 0][::-1]
    # ahead_right_tmp=aheadRow[:]
    # ahead_right_tmp[:, 0]=ahead_right_tmp[:, 0]-45

    # middle_left_tmp=middleRow[:]
    # middle_left_tmp[:,0]=ahead_right_tmp[:, 0]-45

    straight_ahead_right = map(
        aheadRow-[45, 0, 0], (-90, 90), (SERVOMIN_R, SERVOMAX_R))
    straight_middle_right = map(middleRow, (-90, 90), (SERVOMIN_R, SERVOMAX_R))
    straight_back_right = map(backRow, (-90, 90), (SERVOMIN_R, SERVOMAX_R))

    straight_ahead_left = map(-1 * aheadRow, (-90, 90),
                              (SERVOMIN_L, SERVOMAX_L))
    straight_middle_left = map(-1 * (middleRow -
                                     [45, 0, 0]), (-90, 90), (SERVOMIN_L, SERVOMAX_L))
    straight_back_left = map(-1 * backRow, (-90, 90), (SERVOMIN_L, SERVOMAX_L))

    straight_ahead_right = straight_ahead_right.astype(np.int32)
    straight_middle_right = straight_middle_right.astype(np.int32)
    straight_back_right = straight_back_right.astype(np.int32)

    straight_ahead_left = straight_ahead_left.astype(np.int32)
    straight_middle_left = straight_middle_left.astype(np.int32)
    straight_back_left = straight_back_left.astype(np.int32)

    with open('arduino.txt', 'w') as f:
        txt_writer(f, straight_ahead_right)
        f.write('\n')
        txt_writer(f, straight_middle_right)
        f.write('\n')
        txt_writer(f, straight_back_right)

        f.write('\n')
        txt_writer(f, straight_ahead_left)
        f.write('\n')
        txt_writer(f, straight_middle_left)
        f.write('\n')
        txt_writer(f, straight_back_left)


def txt_writer(f, data):
    f.write('{')
    for k, elem in enumerate(data):
        f.write('{')
        for j, element in enumerate(elem):
            f.write(str(element))
            if j != 2:
                f.write(',')
        f.write('}')
        if k != 49:
            f.write(',')
    f.write('}')

# def get_turn_data():
#     generate_use_data_t()


def map(input_num, input_range, output_range):
    a = (output_range[1]-output_range[0])/(input_range[1]-input_range[0])
    b = output_range[0]-input_range[0]*a
    result = a*input_num+b
    return result


if __name__ == "__main__":
    # label = np.load('./data/label_ahead.npy')
    # data = np.load('./data/data_ahead.npy')
    # # model=build_model()
    # model=model_2()
    # label = label.reshape(label.shape[0], 1)
    # history = model.fit(
    #     label, data, epochs=150, batch_size=8, validation_split=0.05)
    # model.save('./data/NN_ahead.h5')
    # graph('ahead')

    # label = np.load('./data/label_middle.npy')
    # data = np.load('./data/data_middle.npy')
    # # # model=build_model()
    # model=model_2()
    # label = label.reshape(label.shape[0], 1)
    # history = model.fit(
    #     label, data, epochs=100, batch_size=8, validation_split=0.1)
    # model.save('./data/NN_middle.h5')
    # graph('middle')

    # label = np.load('./data/label_back.npy')
    # data = np.load('./data/data_back.npy')
    # # # model=build_model()
    # model=model_2()
    # label = label.reshape(label.shape[0], 1)
    # history = model.fit(
    #     label, data, epochs=150, batch_size=8, validation_split=0.05)
    # model.save('./data/NN_back.h5')
    # graph('back')

    # graph('back')
    # graph('middle')
    # graph('ahead')
    # -10 - 100 55 - -55
    # generate_use_data([[-12,25],[-30,30],[-25,12]],25)
    # generate_use_data_t([[22,9],[13,-13],[-9,-22]],25)
    # generate_use_data_t([[22,8],[7,-7],[-8,-22]],25)
    # generate_use_data_t([[20.35, 5.10], [9, -6.25], [-5.10, -20.35]],30)
    # generate_use_data_t([[20.35, 5.10], [9, -6.25], [-5.10, -20.35]],30)
    # generate_use_data_t([[26.5, -5], [16, -16], [5, -26.5]],30)
    # data={'delay_time': 0.001985700066062273, 'length': 13.672688163296883, 'st_point_a': 22.108572000720343, 'st_point_m': -12.868432012248148}
    # a={"delay_time": 0.003983117520340619, "length": 16.597289753255783, "st_point_a": 23.011340102173662, "st_point_m": -3.3471054371900393}
    # a = {"delay_time": 0.003623675222260717, "length": 15.249794309080531,
    #      "st_point_a": 20.35957752716881, "st_point_m": -6.258786473403007}  # TODO fasttest
    # a = {"delay_time": 0.003623675222260717, "length": 1,
    #      "st_point_a": 20.35957752716881, "st_point_m": -6.258786473403007}  # TODO fro turn 
    # a={'delay_time': 0.0038839442858743124,'length': 15.02214366569923,'st_point_a': 21.483755179408615,'st_point_m': -6.159917245448711}
    # a={"delay_time": 0.013257107652327172, "length": 24.115631457710805, "st_point_a": 26.187273919059535, "st_point_m": -12.171231015717657} #TODO fastest   4.88s/1.6m
    # a={'delay_time': 0.010778506064900967,'length': 23.39719477061695,'st_point_a': 25.870626368447393,'st_point_m': -12.080083913592235}
    # generate_use_data_t([[a['st_point_a'], a['st_point_a']-a['length']], [a['st_point_m'] +a['length'], a['st_point_m']], [-1*a['st_point_a']+a['length'], -1*a['st_point_a']]], 30)
    # gen_data_for_arduino()
    # gen_test()
    pass
