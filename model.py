from tensorflow.contrib import rnn
import tensorflow as tf

class LstmModel():

   def __init__(self, sess, name, data_dim, hidden_dim, output_dim, seq_length, stack_size, learning_rate):
       self.sess = sess
       self.name = name
       self.data_dim = data_dim
       #lstm cell memory
       self.hidden_dim = hidden_dim
       self.output_dim = output_dim
       self.seq_length = seq_length
       self.stack_size = stack_size

       self.learning_rate = learning_rate

       self._build_net()


   def _build_net(self):
       with tf.variable_scope(self.name):

           # input place holders
           self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.data_dim])
           self.Y = tf.placeholder(tf.float32, [None, 1])

           # Make a lstm cell with hidden_size (each unit output vector size)
           def lstm_cell():
               cell = rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
               return cell

           cell = rnn.MultiRNNCell([lstm_cell() for _ in range(self.stack_size)], state_is_tuple=True)

           outputs, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
           self.Y_pred = tf.contrib.layers.fully_connected(
               outputs[:, -1], self.output_dim, activation_fn=None)  # We use the last cell's output

           # RMSE
           self.targets = tf.placeholder(tf.float32, [None, 1])
           self.predictions = tf.placeholder(tf.float32, [None, 1])
           self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))
           self.accuracy_summ = tf.summary.scalar("accuracy", self.rmse)

       with tf.name_scope("cost") as scope:
           self.loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares
           self.cost_summ = tf.summary.scalar("cost", self.loss)

       with tf.name_scope("train") as scope:
           # optimizer
           self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
           self.train = self.optimizer.minimize(self.loss)

       with tf.name_scope("tensorboard") as scope:
           # tensorboard --logdir=./logs/lstm_logs_r0_01
           self.merged_summary = tf.summary.merge_all()
           writer = tf.summary.FileWriter("./logs/lstm_logs_r0_01")
           writer.add_graph(self.sess.graph)  # Show the graph


   #def predict(self, x_test, keep_prop=1.0):
       #return self.sess.run(self.logits,
       #     feed_dict={self.X: x_test, self.keep_prob: keep_prop})

   def model_predict(self, x_test):
       return self.sess.run(self.Y_pred, feed_dict={self.X: x_test})

    #def get_accuracy(self, x_test, y_test, keep_prop=1.0):
    #   return self.sess.run(self.accuracy,
    #    feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})


   def get_accuracy(self, predict, y_test):
       return self.sess.run(self.rmse, feed_dict={targets: y_test, predictions: predict})


    # def train(self, x_data, y_data, keep_prop=0.7):
        #return self.sess.run([self.cost, self.optimizer], feed_dict={
       #    self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

   def excute_train(self, x_data, y_data):
       return self.sess.run([self.merged_summary, self.loss, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})
