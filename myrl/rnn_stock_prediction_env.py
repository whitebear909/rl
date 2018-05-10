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



# Standardization
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()


# 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
# Min-Max scaling
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)  # 1e-7은 0으로 나누는 오류 예방차원


# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


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

    #데이터의위치변경진행
    close = xy[:, [-1]] #close copy
    temp = np.delete(xy, 4, 1)
    volume = temp[:, [-1]] #volume copy
    temp = np.delete(xy, 3, 1) #delete volume
    price = np.concatenate((temp, close), axis=1)  # axis=1, 세로로 합친다

    norm_price = min_max_scaling(price)  # 가격형태 데이터 정규화 처리
    print("price.shape: ", price.shape)
    print("price[0]: ", price[0])
    print("norm_price[0]: ", norm_price[0])
    print("=" * 100)  # 화면상 구분용

    norm_volume = min_max_scaling(volume)  # 거래량형태 데이터 정규화 처리
    print("volume.shape: ", volume.shape)
    print("volume[0]: ", volume[0])
    print("norm_volume[0]: ", norm_volume[0])
    print("=" * 100)  # 화면상 구분용
    #x = np.concatenate((norm_price, norm_volume), axis=1)  # axis=1, 세로로 합친다

    xy = np.concatenate((norm_price, norm_volume), axis=1)  # axis=1, 세로로 합친다

    #xy = MinMaxScaler(xy)
    x = xy
    y = xy[:, [-2]]  # Close as label

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

    # sequence length만큼의 가장 최근 데이터를 슬라이싱한다
    recent_data = np.array([xy[len(x) - seq_length:]])
    print("recent_data.shape:", recent_data.shape)
    print("recent_data:", recent_data)

    # 내일 종가를 예측해본다
    test_predict = model.run_evaluate(recent_data)

    print("test_predict", test_predict[0])
    test_predict = reverse_min_max_scaling(price, test_predict)  # 금액데이터 역정규화한다
    print("Tomorrow's stock price", test_predict[0])  # 예측한 주가를 출력한다