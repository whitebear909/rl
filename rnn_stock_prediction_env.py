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


    '''
    데이터를 생성한다
    '''
    # Open, High, Low, Volume, Close
    #env = DailyTradingEnv()
    #xy = env.tic_que

    xy = np.loadtxt('./data-02-stock_daily.csv', delimiter=',')
    xy = xy[::-1]  # reverse order (chronically ordered)

    xy = MinMaxScaler(xy)
    x = xy
    y = xy[:, [-1]]  # Close as label

    # build a dataset
    dataX = []
    dataY = []
    for i in range(0, len(y) - seq_length):
        _x = x[i:i + seq_length]
        _y = y[i + seq_length]  # Next close price
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)

    # train/test split
    train_size = int(len(dataY) * 0.7)
    test_size = len(dataY) - train_size
    trainX, testX = np.array(dataX[0:train_size]), np.array(
        dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(
        dataY[train_size:len(dataY)])


    n_layers = 3
    cell_units = 128
    nb_epoch = 500
    batch_size = 1
    p_keep = 1.0
    p_learning_rate = 0.001
    n_training_stock = 200

    '''
    모델을 설정한다
    #lstm 3 stack, other 3stack, 100 w
    '''
    model = LstmModelF("lstm", len(trainX[0][0]), len(trainY[0]), seq_length,
                       n_layers, cell_units)

    '''    
    모델을 학습시킨다
    '''
    model.fit(trainX, trainY,
              nb_epoch,
              train_size,
              p_keep,
              p_learning_rate)

    '''
    예측 정확도를 평가한다
    '''
    accuracy = model.evaluate(testX, testY, test_size)
    #print('accuracy: ', accuracy)

    # Plot predictions
    plt.plot(testY)
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