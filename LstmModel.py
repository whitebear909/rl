from tensorflow.contrib import rnn
import tensorflow as tf


class LstmModelF():

    def __init__(self, name, n_in, n_out, n_seq, n_hiddens, learning_rate):
        self.name = name

        # lstm cell memory
        #self.cell_units = 128
        self.cell_units = 12
        self.n_seq = n_seq

        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.n_out = n_out
        self.weights = []
        self.biases = []

        self.learning_rate = learning_rate

        self._x = None
        self._t = None
        self._ylist = None
        self._keep_prob = None
        self._sess = None
        self._history = {
            'rmse': [],
            'accuracy': [],
            'loss': []
        }

    def inference(self, x, keep_prob):

        # Make a lstm cell with hidden_size (each unit output vector size)
        def lstm_cell():
            cell = rnn.BasicLSTMCell(self.cell_units, state_is_tuple=True)
            return cell

        cell = rnn.MultiRNNCell([lstm_cell() for _, _ in enumerate(self.n_hiddens)], state_is_tuple=True)

        outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

        y = tf.contrib.layers.fully_connected(
            outputs[:, -1], self.n_out, activation_fn=None)  # We use the last cell's output

        return y

    def loss(self, y, t):

        cross_entropy = tf.reduce_sum(tf.square(t - y))  # sum of the squares
        self.cost_summ = tf.summary.scalar("cost", cross_entropy)

        return cross_entropy

    def trainging(self, loss):

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_step = optimizer.minimize(loss)

        return train_step

    def accuracy(self, y, t):
        accuracy = tf.sqrt(tf.reduce_mean(tf.square(y - t)))
        return accuracy
        #rmse = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.t)))
        #accuracy = self._sess.run(self.rmse, feed_dict={self._targets: y, self._predictions: t})
        # self.accuracy_summ = tf.summary.scalar("accuracy", self.rmse)

        # with tf.name_scope("tensorboard") as scope:
        # tensorboard --logdir=./logs/lstm_logs_r0_01

    def evaluate(self, X_test, Y_test):
        self.result_y = self._y.eval(session=self._sess,feed_dict={
            self._x: X_test
        })

        accuracy = self.accuracy(self._y, self._t)
        return accuracy.eval(session=self._sess, feed_dict={
            self._x: X_test,
            self._t: Y_test
            # RNN is not need keep_prob
            # , self._keep_prob: 1.0
        })
    

    def fit(self, X_train, Y_train,
            epochs=100, batch_size=100, p_keep=0.5,
            verbose=1):
        x = tf.placeholder(tf.float32, [None, self.n_seq, self.n_in])
        t = tf.placeholder(tf.float32, [None, self.n_out])
        keep_prob = tf.placeholder(tf.float32)

        # evaluate()
        self._x = x
        self._t = t
        self._keep_prob = keep_prob

        y = self.inference(x, keep_prob)
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

            # record values
            self._history['loss'].append(loss_)
            self._history['accuracy'].append(accuracy_)

            if verbose:
                print('epoch:', epoch,
                      ' loss:', loss_,
                      ' accuracy:', accuracy_)

        return self._history

    def save_model(self):
        self.save_path = self.saver.save(self.sess, "./lstm.ckpt")

    def load_model(self):
        # ckpt = tf.train.get_checkpoint_state(self.name)
        # self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        # new_saver = tf.train.import_meta_graph("./lstm.ckpt.meta")
        self.saver.restore(self.sess, "./lstm.ckpt")
