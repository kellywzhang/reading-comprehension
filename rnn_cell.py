"""
Goal:
    - Create RNN cells

TODO/ISSUES: Initilization, Testing, zero-state, batch dimension?, Check dimensions on EVERYTHING

Credits: Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py
"""

import TensorFlow as tf

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

class RNNCell(object):
  """Abstract object representing an RNN cell."""

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state."""
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s)."""
    return tf.zeros(shape=[batch_size, self.state_size])

class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, state_size, input_size, activation=tanh):
    self._state_size = state_size
    self._output_size = state_size
    self._input_size = input_size
    self._activation = activation

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with state_size dimension cells."""
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        W_reset = tf.get_variable(name="reset_weight", shape=[state_size, state_size+input_size], \
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        W_update = tf.get_variable(name="update_weight", shape=[state_size, state_size+input_size], \
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b_reset = tf.get_variable(name="reset_bias", shape=[state_size], initializer=tf.constant_initializer(0.0))
        b_update = tf.get_variable(name="update_bias", shape=[state_size], initializer=tf.constant_initializer(0.0))

        reset = sigmoid(tf.matmul(W_reset, inputs) + b_reset)
        update = sigmoid(tf.matmul(W_update, inputs) + b_update)

      with vs.variable_scope("Candidate"):
        W_candidate = tf.get_variable(name="candidate_weight", shape=[state_size, state_size+input_size], \
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b_candidate = tf.get_variable(name="candidate_bias", shape=[state_size], \
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

        reset_input = tf.concat(0, [reset * state, inputs]) #check dimension, batches?
        candidate = self._activation(tf.matmul(W_reset, reset_input) + b_candidate)

      new_h = update * state + (1 - update) * candidate
    return new_h, new_h

    def zero_state():
        return tf.Variable(tf.zeros([state_size]))

class MultiRNNCell(RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells):
    """Create a RNN cell composed sequentially of a number of RNNCells."""
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    self._cells = cells

  @property
  def state_size(self):
      return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):
            cur_state = array_ops.slice(
                state, [0, cur_state_pos], [-1, cell.state_size])
            cur_state_pos += cell.state_size
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    new_states = (array_ops.concat(1, new_states))
    return cur_inp, new_states
