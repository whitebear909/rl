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
    #nb_epoch = 500
    nb_epoch = 1
    batch_size = 1
    p_keep = 1.0
    p_learning_rate = 0.001
    n_training_stock = 200
    predict_days = 5

    env = DailyTradingEnv(seq_length, test_date_rate, data_type, file_path)

    '''
    모델을 설정한다
    '''
    '''
    models = [LstmModelF(str(i), len(env.trainX[0][0]), 1, seq_length,
                       n_layers, cell_units) for i in range(0, len(env.trainX[0][0]))]
    '''
    models = []
    for i in range(0,6):
        models.append(LstmModelF(str(i), len(env.trainX[0][0]), 1, seq_length, n_layers, cell_units))

    '''    
    모델을 학습시킨다
    '''
    #for i, model in enumerate(models):
    for i in range(0,6):
        models[i].fit(env.trainX, env.trainY[:, [i]],
                  nb_epoch,
                  env.train_size,
                  p_keep,
                  p_learning_rate)
        tf.reset_default_graph()
    '''    
    모델을 평가한다
    '''
    for i, model in enumerate(models):
        accuracy[i] = model.evaluate(env.testX, env.testY[:, [i]], env.test_size)
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

    for i in range(0,predict_days):
        if i==0:
            recent_data = env.get_recent_data()
        else :
            recent_data = test_predict[i-1]
        print("recent_data.shape:", recent_data.shape)
        print("recent_data:", recent_data)
        for j, model in enumerate(models):
            # 내일을 예측해본다
            test_predict[i][j] = model.run_evaluate(recent_data)
            print("test_predict", test_predict[i][j][0])
            if j==3:
                test_predict_rmxc[i][j] = env.reverse_min_max_scaling(env.volume, test_predict[i][j])  # 금액데이터 역정규화한다
            else:
                test_predict_rmxc[i][j] = env.reverse_min_max_scaling(env.price, test_predict[i][j])  # 금액데이터 역정규화한다
            print("Tomorrow's ?", test_predict_rmxc[i][j][0])  # 예측한 값을 출력한다

    # Open, High, Low, Volume, Close
    '''    
    model = LstmModelF("CLOSE", len(env.trainX[0][0]), 1, seq_length,
                       n_layers, cell_units)
    model.fit(env.trainX, env.trainY[:,[3]],
              nb_epoch,
              env.train_size,
              p_keep,
              p_learning_rate)
              
    accuracy = model.evaluate(env.testX, env.testY[:,[3]], env.test_size)          
    '''


    '''
    #print('accuracy: ', accuracy)
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
    print("Tomorrow's stock price", test_predict[0])  # 예측한 주가를 출력한다
    '''