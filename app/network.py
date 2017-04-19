import tensorflow as tf
import numpy as np
from time_helper import TimeHelper
import random
import os.path
import time

CHECKPOINT_DIR = 'saved/'

class Network:

  def __init__(self, data, lstm_size=128, num_layers=2, learning_rate=0.003, lr_decay=0.9, momentum=0.0, epsilon=1e-10, forget_bias=1.0, std_dev_init=0.01, name="rnn", ckpt_file='model.ckpt'):
    self.scope = name
    self.sess = None
    self.saver = None

    self.data = data
    self.in_size = self.out_size = len(self.data.get_vocab())
    self.lstm_size = lstm_size
    self.num_layers = num_layers
    self.ckpt_file = CHECKPOINT_DIR + ckpt_file;

    self.learning_rate = tf.constant( learning_rate )

    # Last state of LSTM, used when running the network in TEST mode
    self.lstm_last_state = np.zeros((self.num_layers*2*self.lstm_size,))

    with tf.variable_scope(self.scope):
      self.x_input = tf.placeholder(tf.float32, shape=(None, None, self.in_size), name="x_input")
      self.lstm_init_value = tf.placeholder(tf.float32, shape=(None, self.num_layers*2*self.lstm_size), name="lstm_init_value")

      # LSTM
      lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=forget_bias, state_is_tuple=False)
      lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=False)

      # Iteratively compute output of recurrent network
      outputs, self.lstm_new_state = tf.nn.dynamic_rnn(lstm, self.x_input, initial_state=self.lstm_init_value, dtype=tf.float32)

      # Linear activation (FC layer on top of the LSTM net)
      rnn_out_W = tf.Variable(tf.random_normal( (self.lstm_size, self.out_size), stddev=std_dev_init ))
      rnn_out_B = tf.Variable(tf.random_normal( (self.out_size, ), stddev=std_dev_init ))

      outputs_reshaped = tf.reshape( outputs, [-1, self.lstm_size] )
      network_output = ( tf.matmul( outputs_reshaped, rnn_out_W ) + rnn_out_B )

      batch_time_shape = tf.shape(outputs)
      self.final_outputs = tf.reshape( tf.nn.softmax( network_output), (batch_time_shape[0], batch_time_shape[1], self.out_size) )


      ## Training: provide target outputs for supervised training.
      self.y_batch = tf.placeholder(tf.float32, (None, None, self.out_size))
      y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])

      self.cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=y_batch_long) )
      self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.99, epsilon=epsilon).minimize(self.cost)
      # self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=lr_decay, momentum=momentum, epsilon=epsilon).minimize(self.cost)

  def get_tensorflow_session(self):
    if not self.sess:
      self.sess = self.sess or tf.InteractiveSession()
      self.sess.run(tf.global_variables_initializer())

    return self.sess

  def get_saver(self):
    if not self.saver:
      self.saver = tf.train.Saver(tf.global_variables())
    return self.saver

  def restore_network(self):
    self.get_saver()
    if os.path.isfile(self.ckpt_file + '.index'):
      self.saver.restore(self.sess, self.ckpt_file)

  def train_batch(self, batch_size, num_batches, time_steps):
    self.get_tensorflow_session()
    self.restore_network();

    th = TimeHelper()

    th.get_current_time_ms()

    batch = np.zeros((batch_size, time_steps, self.in_size))
    batch_y = np.zeros((batch_size, time_steps, self.in_size))

    possible_batch_ids = range(self.data.get_embedded_data().shape[0] - time_steps - 1)
    for i in range(num_batches):
      batch_ids = random.sample(possible_batch_ids, batch_size)

      for j in range(time_steps):
        char_index = [k+j for k in batch_ids]
        next_char_index= [k+j+1 for k in batch_ids]

        embedded_data = self.data.get_embedded_data()
        
        batch[:, j, :] = embedded_data[char_index, :]
        batch_y[:, j, :] = embedded_data[next_char_index, :] 

      init_value = np.zeros((batch_size, self.num_layers*2*self.lstm_size))
      cost, _ = self.sess.run([self.cost, self.train_op], feed_dict={self.x_input:batch, self.y_batch:batch_y, self.lstm_init_value:init_value})

      if (i + 1) % 100 == 0 and i != 0:
        new_time_ms = th.get_current_time_ms()
        time_between_batches_ms = th.diff_ms(new_time_ms);
        batches_per_second = round(100/th.ms_to_s(time_between_batches_ms), 3)
        batches_left = num_batches - i
        etl =  th.s_format(round(batches_left / batches_per_second, 3))

        print "batch: %s, loss: %s; batches per second: %s; estimated time left: %s" % (i+1, cost, batches_per_second, etl)

      if (i + 1) % 1000 == 0 and i != 0:
        self.saver.save(self.sess, self.ckpt_file)
        print self.get_sentence()

  def get_sentence(self, start_string="the", ending_values=['.', '?', '!']):
    self.get_tensorflow_session()
    # Not sure if this is the best way to do this. We may want to check if the network is already initialized from training.
    self.restore_network()

    sentence = start_string.lower();
    limit = None
    if type(ending_values) is int:
      limit = ending_values
      ending_values = None;
    else:
      limit = 100

    embedded_start_string = self.data.embed(start_string)
    self.lstm_last_state = np.zeros((self.num_layers * 2 * self.lstm_size,))
    for embedded_char in embedded_start_string:
      output_probabilities = self.run_step([embedded_char])

    for i in range(limit):
      vocab_index = np.random.choice(range(len(self.data.get_vocab())), p=output_probabilities)
      next_char = self.data.get_char_from_embed(vocab_index)
      sentence += next_char
      output_probabilities = self.run_step(self.data.embed(next_char))
      if ending_values and next_char in ending_values:
        break;

    return sentence


  def run_step(self, input):
    output, lstm_state = self.sess.run([self.final_outputs, self.lstm_new_state], feed_dict={self.x_input:[input], self.lstm_init_value:[self.lstm_last_state]})
    self.lstm_last_state = lstm_state[0]
    return output[0][0]
