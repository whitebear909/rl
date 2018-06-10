from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np


class LstmModelF():

    def __init__(self, name, n_in, n_out, n_seq, n_layers, cell_units, keep_prob):
        self.name = name

        # lstm cell memory
        self.cell_units = cell_units
        self.n_seq = n_seq

        self.n_in = n_in
        self.number_of_layers = n_layers
        self.n_out = n_out
        self.result_y = []
        self.forget_bias = 1.0  # 망각편향(기본값 1.0)
        self.keep_prob = keep_prob

        # 학습에 직접적으로 사용하지 않고 학습 횟수에 따라 단순히 증가시킬 변수를 만듭니다.
        self._global_step = None
        self._x = None
        self._t = None
        self._ylist = None
        self._learning_rate = None
        self._keep_prob = None
        self._batch_size = None
        self._sess = None
        self._state = None

        self._history = {
            'rmse': [],
            'accuracy': [],
            'loss': []
        }

        self.make_model()

    def inference(self, x, batch_size, keep_prob):
        # Make a lstm cell with hidden_size (each unit output vector size)
        def lstm_cell():
            # LSTM셀을 생성
            # num_units: 각 Cell 출력 크기
            # forget_bias:  to the biases of the forget gate
            #              (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
            # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
            # state_is_tuple: False ==> they are concatenated along the column axis.
            cell = rnn.BasicLSTMCell(num_units=self.cell_units,
                                     forget_bias=self.forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
            if self.keep_prob < 1.0:
                cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        # cell = rnn.MultiRNNCell([lstm_cell() for _, _ in enumerate(self.n_hiddens)], state_is_tuple=True)
        stacked_cell = rnn.MultiRNNCell(
            [lstm_cell() for _ in range(self.number_of_layers)], state_is_tuple=True)

        # print(tf.shape(x))
        # print(tf.shape(x)[0])
        self._cell = stacked_cell
        self._initial_state = stacked_cell.zero_state(batch_size, tf.float32)

        outputs, _states = tf.nn.dynamic_rnn(stacked_cell, x, dtype=tf.float32, initial_state=self._initial_state)
        self._state = _states
        # [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
        # 과거 여러 거래일의 주가를 이용해서 다음날의 주가 1개를 예측하기때문에 MANY-TO-ONE형태이다
        y = tf.contrib.layers.fully_connected(
            outputs[:, -1], self.n_out, activation_fn=tf.identity)  # We use the last cell's output

        return y

    def loss(self, y, t):

        cross_entropy = tf.reduce_sum(tf.square(t - y))  # sum of the squares
        self.cost_summ = tf.summary.scalar("cost", cross_entropy)

        return cross_entropy

    def trainging(self, loss, learning_rate):

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(loss, global_step=self._global_step)

        return train_step

    def accuracy(self, y, t):
        accuracy = tf.sqrt(tf.reduce_mean(tf.square(y - t)))
        return accuracy
        # rmse = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.t)))
        # accuracy = self._sess.run(self.rmse, feed_dict={self._targets: y, self._predictions: t})
        # self.accuracy_summ = tf.summary.scalar("accuracy", self.rmse)

        # with tf.name_scope("tensorboard") as scope:
        # tensorboard --logdir=./logs/lstm_logs_r0_01

    def evaluate(self, X_test, Y_test):

        xtest = np.reshape(X_test[0], [-1, 10, 5])
        ytest = np.reshape(Y_test[0], [-1, 1])

        batch_size = len(ytest)
        state = self._sess.run(self._cell.zero_state(batch_size, tf.float32))
        test_data_feed = {
            self._x: xtest,
            self._t: ytest,
            self._batch_size: batch_size,
            self._learning_rate: 0.0,
            self._keep_prob: 1.0,
            self._initial_state: state
        }
        self.result_y = self._y.eval(session=self._sess, feed_dict=test_data_feed)
        accuracy = self.accuracy(self._y, self._t)
        return accuracy.eval(session=self._sess, feed_dict=test_data_feed)

    def run_evaluate(self, X_test):

        p_batch_size = 1.0
        self.keep_prob = 1.0
        state = self._sess.run(self._cell.zero_state(p_batch_size, tf.float32))

        test_data_feed = {
            self._x: X_test,
            self._batch_size: p_batch_size,
            self._learning_rate: 0.0,
            self._keep_prob: 1.0,
            self._initial_state: state
        }

        self.result_y = self._y.eval(session=self._sess, feed_dict=test_data_feed)

        return self.result_y

    def make_model(self):
        self._global_step = tf.Variable(0, trainable=False, name='global_step')
        self._x = tf.placeholder(tf.float32, [None, self.n_seq, self.n_in])
        self._t = tf.placeholder(tf.float32, [None, self.n_out])
        self._keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")
        self._learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        self._batch_size = tf.placeholder(tf.int32, [], name="batch_size")

        self._y = self.inference(self._x, self._batch_size, self._keep_prob)
        self._loss = self.loss(self._y, self._t)
        self._train_step = self.trainging(self._loss, self._learning_rate)
        self._accuracy = self.accuracy(self._y, self._t)

        self._sess = tf.Session()
        self._saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state('./savemodel')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            self._sess.run(tf.global_variables_initializer())


    def fit(self, X_train, Y_train, global_epochs, epochs, p_learning_rate, verbose=1):
        for g in range(0, global_epochs):
            for i in range(0, len(Y_train)):

                xtrain = np.reshape(X_train[i],[-1,10,5])
                ytrain = np.reshape(Y_train[i],[-1,1])

                batch_size = len(ytrain)

                state = self._sess.run(self._cell.zero_state(batch_size, tf.float32))

                for epoch in range(epochs):

                    train_data_feed = {
                        self._x: xtrain,
                        self._t: ytrain,
                        self._batch_size: batch_size,
                        self._learning_rate: p_learning_rate,
                        self._keep_prob: self.keep_prob,
                        self._initial_state: state
                    }

                    self._sess.run(self._train_step, feed_dict=train_data_feed)
                    loss_ = self._loss.eval(session=self._sess, feed_dict=train_data_feed)
                    accuracy_ = self._accuracy.eval(session=self._sess, feed_dict=train_data_feed)

                    self._history['loss'].append(loss_)
                    self._history['accuracy'].append(accuracy_)

                if verbose:
                    print('g_epoch:', g,
                          'i_stock:', i,
                          #' epoch:', epoch,
                          ' loss:', loss_,
                          ' rmse:', accuracy_,
                          ' step:', self._sess.run(self._global_step)
                          )

            # 최적화가 끝난 뒤, 변수를 저장합니다.
            self._saver.save(self._sess, './savemodel/lstm.ckpt', global_step=self._global_step)
        return self._history

    def save_model(self):

        self.save_path = self._saver.save(self._sess, "./savemodel/lstm.ckpt")

#    def load_model(self):


# ckpt = tf.train.get_checkpoint_state(self.name)
# self.saver.restore(self.sess, ckpt.model_checkpoint_path)
# new_saver = tf.train.import_meta_graph("./lstm.ckpt.meta")
# self.saver.restore(self.sess, "./lstm.ckpt")