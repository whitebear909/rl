'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf

import numpy as np
import matplotlib
import os
from env import *
from model.LstmModel import *


tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


if __name__ == '__main__':

    # train Parameters
    seq_length = 10
    test_date_rate = 0.7
    data_type = 'File'
    file_path = './data-02-stock_daily.csv'
    n_layers = 3
    cell_units = 128
    nb_epoch = 500
    batch_size = 1
    p_keep = 1.0
    p_learning_rate = 0.001
    n_training_stock = 200

    env = DailyTradingEnv(seq_length, test_date_rate, data_type, file_path)
    '''
    모델을 설정한다
    #lstm 3 stack, other 3stack, 100 w
    '''
    model = LstmModelF("lstm", len(env.trainX[0][0]), len(env.trainY[0]), seq_length,
                       n_layers, cell_units)

    '''    
    모델을 학습시킨다
    '''
    model.fit(env.trainX, env.trainY,
              nb_epoch,
              env.train_size,
              p_keep,
              p_learning_rate)

    '''
    예측 정확도를 평가한다
    '''
    accuracy = model.evaluate(env.testX, env.testY, env.test_size)
    #print('accuracy: ', accuracy)

    # Plot predictions
    plt.plot(env.testY)
    plt.plot(model.result_y)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()

    fig = plt.figure()
    ax_acc = fig.add_subplot(111) #axis accuracy
    ax_acc.plot(range(nb_epoch),model._history['accuracy'], label='rmse', color='black')

    ax_acc = ax_acc.twinx()  # axis loss
    ax_acc.plot(range(nb_epoch), model._history['loss'], label='loss', color='red')

    plt.xlabel("epoch")
    plt.show()
    #plt.savefig('./plt/lstm.eps')

    # sequence length만큼의 가장 최근 데이터를 슬라이싱한다
    recent_data = env.get_recent_data()
    print("recent_data.shape:", recent_data.shape)
    print("recent_data:", recent_data)

    # 내일 종가를 예측해본다
    test_predict = model.run_evaluate(recent_data)

    print("test_predict", test_predict[0])
    test_predict = env.reverse_min_max_scaling(env.price, test_predict)  # 금액데이터 역정규화한다
    print("Tomorrow's stock price", test_predict[0])  # 예측한 주가를 출력한다