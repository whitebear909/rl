'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf

import numpy as np
import matplotlib
import os
from env import *
from model.model_upgrade import *


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
    learning_rate = 0.01
    n_hiddens = [100, 100, 100]
    nb_epoch = 500
    batch_size = 200
    p_keep = 0.5



    '''
    데이터를 생성한다
    '''
    # Open, High, Low, Volume, Close
    # env = DailyTradingEnv()
    # xy = env.tic_que

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


    #print (len(trainX[0][0]))
    #print(len(trainX[0]),len(trainX[1]),len(trainX[2]),len(trainX[0][-1]))

    '''
    모델을 설정한다
    #lstm 3 stack, other 3stack, 100 w
    '''
    model = LstmModelF("lstm", len(trainX[0][0]), len(trainY[0]), seq_length, n_hiddens, learning_rate)

    '''    
    모델을 학습시킨다
    '''
    model.fit(trainX, trainY,
              nb_epoch,
              batch_size,
              p_keep)

    '''
    예측 정확도를 평가한다
    '''
    accuracy = model.evaluate(testX, testY)
    print('accuracy: ', accuracy)


