"""
Goal:
    - Create RNN cells

TODO/ISSUES: Initilization, Testing

Credits: Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py
"""

import tensorflow as tf
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

class RNNCell(object):
  """Abstract object representing an RNN cell."""

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state."""
    raise NotImplementedError("Abstract method")



  def zero_state(self, batch_size):
    """Return zero-filled state tensor(s)."""
    return tf.zeros(shape=[batch_size, self._state_size])

class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, state_size, input_size, activation=tanh):
    self._state_size = state_size
    self._output_size = state_size
    self._input_size = input_size
    self._activation = activation

  # Pass in mask for different length sequences
  def __call__(self, inputs, state, scope=None):
  #def __call__(self, inputs, state, time_mask, scope=None):
    """Gated recurrent unit (GRU) with state_size dimension cells."""
    with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
        input_size = self._input_size
        state_size = self._state_size

        hidden = tf.concat(1, [state, inputs])

        with tf.variable_scope("Gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            W_reset = tf.get_variable(name="reset_weight", shape=[state_size+input_size, state_size], \
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            W_update = tf.get_variable(name="update_weight", shape=[state_size+input_size, state_size], \
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            b_reset = tf.get_variable(name="reset_bias", shape=[state_size], initializer=tf.constant_initializer(1.0))
            b_update = tf.get_variable(name="update_bias", shape=[state_size], initializer=tf.constant_initializer(1.0))

            reset = sigmoid(tf.matmul(hidden, W_reset) + b_reset)
            update = sigmoid(tf.matmul(hidden, W_update) + b_update)

        with tf.variable_scope("Candidate"):
            W_candidate = tf.get_variable(name="candidate_weight", shape=[state_size+input_size, state_size], \
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            b_candidate = tf.get_variable(name="candidate_bias", shape=[state_size], \
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

            reset_input = tf.concat(1, [reset * state, inputs])
            candidate = self._activation(tf.matmul(reset_input, W_reset) + b_candidate)

        new_h = update * state + (1 - update) * candidate
        
        #anti_time_mask = tf.cast(time_mask==0, tf.int32)
        #new_h = time_mask * new_h + anti_time_mask * state

    return new_h, new_h

    def zero_state(self, batch_size):
        return tf.Variable(tf.zeros([batch_size, state_size]), dtype=tf.float32)
