from tensorflow.contrib import rnn
import tensorflow as tf


class LstmModelF():

    def __init__(self, name, n_in, n_out, n_seq, n_hiddens, learning_rate):
        self.name = name

        # lstm cell memory
        self.cell_units = 128
        self.n_seq = n_seq

        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.n_out = n_out
        self.weights = []
        self.biases = []

        self.learning_rate = learning_rate

        self._x = None
        self._t = None,
        self._keep_prob = None
        self._sess = None
        self._history = {
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

        '''
        for i, n_hidden in enumerate(self.n_hiddens):
            if i == 0:
                input = x
                input_dim = self.n_in
            else:
                input = output
                input_dim = self.n_hiddens[i-1]
 
            self.weights.append(self.weight_variable([input_dim,n_hidden]))
            self.biases.append(self.bias_variable([n_hidden]))
 
            h = tf.nn.relu(tf.matmul(
                input, self.weights[-1]) + self.biases[-1])
            output = tf.nn.dropout(h,keep_prob)
 
 
        #hidden-out
        self.weights.append(self.weight_variable([self.n_hiddens[-1], self.n_out]))
        self.biases.append(self.bias_variable([self.n_out]))
 
        y=tf.nn.softmax(tf.matmul(
            output, self.weights[-1])+self.biases[-1])
 
        '''
        return y

    def loss(self, y, t):
        '''
        cross_entropy = \
           tf.reduce_mean(-tf.reduce_sum(
                t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                reduction_indices=[1]))
        '''
        cross_entropy = tf.reduce_sum(tf.square(t - y))  # sum of the squares
        self.cost_summ = tf.summary.scalar("cost", cross_entropy)

        return cross_entropy

    def trainging(self, loss):
        '''
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_step = optimizer.minimize(loss)
        '''
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_step = optimizer.minimize(loss)

        return train_step

    def accuracy(self, y, t):

        #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))
        accuracy = self.sess.run(self.rmse, feed_dict={self.targets: y, self.predictions: t})
        # self.accuracy_summ = tf.summary.scalar("accuracy", self.rmse)


        return accuracy

        # with tf.name_scope("tensorboard") as scope:
        # tensorboard --logdir=./logs/lstm_logs_r0_01

    '''
    def get_accuracy(self, predict, y_test):
        return self.sess.run(self.rmse, feed_dict={self.targets: y_test, self.predictions: predict})
    '''

    def evaluate(self, X_test, Y_test):
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

    # def get_accuracy(self, x_test, y_test, keep_prop=1.0):
    #   return self.sess.run(self.accuracy,
    #    feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})
    # def train(self, x_data, y_data, keep_prop=0.7):
    # return self.sess.run([self.cost, self.optimizer], feed_dict={
    #    self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})
    # def predict(self, x_test, keep_prop=1.0):
    # return self.sess.run(self.logits,
#     feed_dict={self.X: x_test, self.keep_prob: keep_prop})