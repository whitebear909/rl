from tensorflow.contrib import rnn
import tensorflow as tf


class LstmModelF():

    def __init__(self, name, n_in, n_out, n_seq, n_layers, cell_units):
        self.name = name

        # lstm cell memory
        # self.cell_units = 128
        self.cell_units = cell_units
        self.n_seq = n_seq

        self.n_in = n_in
        self.number_of_layers = n_layers
        self.n_out = n_out
        self.weights = []
        self.biases = []

        #self.learning_rate = learning_rate
        self.forget_bias = 1.0  # 망각편향(기본값 1.0)

        self._x = None
        self._t = None
        self._ylist = None
        self._learning_rate = None
        self._keep_prob = None
        self._sess = None

        self._history = {
            'rmse': [],
            'accuracy': [],
            'loss': []
        }

    def inference(self, x):
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
            if self._keep_prob < 1.0:
                cell = rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)

            return cell

        #cell = rnn.MultiRNNCell([lstm_cell() for _, _ in enumerate(self.n_hiddens)], state_is_tuple=True)
        stacked_cell = rnn.MultiRNNCell(
            [lstm_cell() for _ in range(self.number_of_layers)], state_is_tuple=True)

        outputs, _states = tf.nn.dynamic_rnn(stacked_cell, x, dtype=tf.float32)

        # [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
        # 과거 여러 거래일의 주가를 이용해서 다음날의 주가 1개를 예측하기때문에 MANY-TO-ONE형태이다
        y = tf.contrib.layers.fully_connected(
            outputs[:, -1], self.n_out, activation_fn=tf.identity)  # We use the last cell's output

        self._saver = tf.train.Saver()
        return y

    def loss(self, y, t):

        cross_entropy = tf.reduce_sum(tf.square(t - y))  # sum of the squares
        self.cost_summ = tf.summary.scalar("cost", cross_entropy)

        return cross_entropy

    def trainging(self, loss):

        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        train_step = optimizer.minimize(loss)

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
        learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        learning_rate = 0.0
        keep_prob = 1.0
        self._learning_rate = learning_rate
        self._keep_prob = keep_prob

        test_data_feed = {
            self._x: X_test,
            self._t: Y_test
        }

        #self.result_y = self._y.eval(session=self._sess, feed_dict={
        #    self._x: X_test
        #})

        self.result_y = self._y.eval(session=self._sess, feed_dict = test_data_feed)

        accuracy = self.accuracy(self._y, self._t)
        return accuracy.eval(session=self._sess, feed_dict=test_data_feed)

    def fit(self, X_train, Y_train,
            epochs=100, batch_size=100, p_keep=0.5, p_learning_rate=0.01,
            verbose=1):
        x = tf.placeholder(tf.float32, [None, self.n_seq, self.n_in])
        t = tf.placeholder(tf.float32, [None, self.n_out])
        keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")
        learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")

        keep_prob = p_keep
        learning_rate = p_learning_rate

        # evaluate()
        self._x = x
        self._t = t
        self._keep_prob = keep_prob
        self._learning_rate = learning_rate

        y = self.inference(x)
        loss = self.loss(y, t)
        train_step = self.trainging(loss)
        accuracy = self.accuracy(y, t)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # evaluate()
        self._y = y
        self._sess = sess

        N_train = len(X_train)
        n_batches = N_train // batch_size

        #initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)

        for epoch in range(epochs):
            # X_, Y_ = suffle(X_train, Y_train)
            # RNN is not need suffle
            X_ = X_train
            Y_ = Y_train

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                sess.run(train_step, feed_dict={
                    x: X_[start:end],
                    t: Y_[start:end]
                    # RNN is not need keep_prob
                    # , keep_prob: p_keep
                })

            loss_ = loss.eval(session=sess, feed_dict={
                x: X_train,
                t: Y_train
                # RNN is not need keep_prob
                # , keep_prob: 1.0
            })

            accuracy_ = accuracy.eval(session=sess, feed_dict={
                x: X_train,
                t: Y_train
                # RNN is not need keep_prob
                # , keep_prob: 1.0
            })

            if epoch % 100 :
                self.save_model()

            # record values
            self._history['loss'].append(loss_)
            self._history['accuracy'].append(accuracy_)

            if verbose:
                print('epoch:', epoch,
                      ' loss:', loss_,
                      ' rmse:', accuracy_)

        return self._history

    def save_model(self):

        self.save_path = self._saver.save(self._sess, "./savemodel/lstm.ckpt")

#    def load_model(self):


# ckpt = tf.train.get_checkpoint_state(self.name)
# self.saver.restore(self.sess, ckpt.model_checkpoint_path)
# new_saver = tf.train.import_meta_graph("./lstm.ckpt.meta")
#self.saver.restore(self.sess, "./lstm.ckpt")