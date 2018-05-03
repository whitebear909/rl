'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf

import numpy as np
import matplotlib
import os
from env import *
from model.model import *


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


# train Parameters
seq_length = 10
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
#trainging_epoch = 3500
trainging_epoch = 500
stack_size = 3
load_model = False

# Open, High, Low, Volume, Close
#env = DailyTradingEnv()
#xy = env.tic_que

xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
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


with tf.Session() as sess:

    md = LstmModel(sess, load_model,  "lstm", data_dim, hidden_dim, output_dim, seq_length, stack_size, learning_rate)

    writer = tf.summary.FileWriter("./logs/lstm_logs_r0_01")
    writer.add_graph(sess.graph)  # Show the graph

    # Training step
    for i in range(trainging_epoch):

        summary, _, step_loss = md.excute_train(trainX, trainY)
        writer.add_summary(summary, global_step=i)
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = md.model_predict(testX)

    # Accuracy step
    rmse_val = md.get_accuracy(test_predict, testY)
    print("RMSE: {}".format(rmse_val))

    #Save model
    md.save_model()

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()