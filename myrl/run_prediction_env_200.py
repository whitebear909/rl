'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf

import numpy as np
import matplotlib
import os
from env_200 import *
from model.LstmModel_200 import *


tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

if __name__ == '__main__':

    # train Parameters
    seq_length = 10
    test_date_rate = 0.7
    data_type = 'DB_Data_Test'
    #train_file_path = './20100101_sample.txt'
    train_file_path = './2018_030200.csv'
    test_file_path = './2018_030200.csv'
    #file_path = './20100101_sample.csv'
    #file_path = './data-02-stock_daily.csv'
    n_layers = 3
    cell_units = 256
    global_epoch = 5
    nb_epoch = 30
    #nb_epoch = 1
    batch_size = 1
    p_keep = 1.0
    p_learning_rate = 0.0001
    n_training_stock = 200
    predict_days = 5

    env = DailyTradingEnv(seq_length, test_date_rate, data_type, train_file_path, test_file_path)

    model = LstmModelF("CLOSE", 5, 1, seq_length, n_layers, cell_units, p_keep)

    #model.fit(env.trainX, env.trainY, global_epoch, nb_epoch, p_learning_rate, 1)

    accuracy = model.evaluate(env.testX, env.testY)

    # Plot predictions
    plt.plot(np.reshape(env.testY,[-1,1]))
    plt.plot(model.result_y)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()

    #need to add multi stocks processing
    '''
    fig = plt.figure()
    ax_acc = fig.add_subplot(111) #axis accuracy
    ax_acc.plot(range(nb_epoch),model._history['accuracy'], label='rmse', color='black')

    ax_acc = ax_acc.twinx()  # axis loss
    ax_acc.plot(range(nb_epoch), model._history['loss'], label='loss', color='red')

    plt.xlabel("epoch")
    plt.show()
    '''

    print('accuracy: ', accuracy)

    # Plot predictions
    #plt.savefig('./plt/lstm.eps')

    # sequence length만큼의 가장 최근 데이터를 슬라이싱한다
    recent_data = env.get_recent_data()
    print("recent_data.shape:", recent_data.shape)
    print("recent_data:", recent_data)

    # 내일 종가를 예측해본다
    test_predict = model.run_evaluate(recent_data)

    print("test_predict", test_predict[0])
    test_predict = env.reverse_min_max_scaling(env.price, test_predict)  # 금액데이터 역정규화한다
    print("Tomorrow's stock price", test_predict[0]) # 예측한 주가를 출력한다