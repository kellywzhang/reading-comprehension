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
                state, [0, cur_state_pos], [-1, cell._state_size])
            cur_state_pos += cell._state_size
        cur_inp, new_state = cell(cur_inp, cur_state)
        new_states.append(new_state)
    new_states = (array_ops.concat(1, new_states))
    return cur_inp, new_states


"""
@property
def state_size(self):
  #size(s) of state(s) used by this cell.
  raise NotImplementedError("Abstract method")

@property
def output_size(self):
  #Integer or TensorShape: size of outputs produced by this cell.
  raise NotImplementedError("Abstract method")

"""






    """seq_len_mask = tf.cast(tf.sequence_mask(seq_lens), tf.float32)
    time = tf.reduce_max(seq_lens)

    def condition(i, inputs, batch_size, embedding_dim, seq_len_mask, state, outputs):
        return tf.less(i, time)

    def body(i, inputs, batch_size, embedding_dim, seq_len_mask, state, outputs):
        with tf.variable_scope("Cell{}".format(i)):
            input_ = tf.slice(inputs, [0, i, 0], [batch_size, 1, embedding_dim])
            time_mask = tf.slice(seq_len_mask, [0, i], [batch_size, 1])
            input_ = tf.squeeze(input_)
            output, state = cell(input_, state, time_mask)
            outputs.append(output)
        return [tf.add(i, 1), inputs, batch_size, embedding_dim, seq_len_mask, state, outputs]

    i = tf.constant(0)
    r = tf.while_loop(condition, body, \
        [i, inputs, batch_size, embedding_dim, seq_len_mask, state, outputs])"""

    """for i in range(5):
        with tf.variable_scope("Cell{}".format(i)):
            input_ = tf.slice(inputs, [0, i, 0], [batch_size, 1, embedding_dim])
            time_mask = tf.slice(seq_len_mask, [0, i], [batch_size, 1])
            input_ = tf.squeeze(input_)
            output, state = cell(input_, state, time_mask)
            outputs.append(output)"""

    #for i in range(tf.reduce_max(seq_lens).eval()):
    """for i in range(5):
        with tf.variable_scope("Cell{}".format(i)):
            input_ = tf.slice(inputs, [0, i, 0], [batch_size, 1, embedding_dim])
            time_mask = tf.slice(seq_len_mask, [0, i], [batch_size, 1])
            input_ = tf.squeeze(input_)
            output, state = cell(input_, state, time_mask)
            outputs.append(output)"""

    #return (outputs, state)

# HAVE TO PAD dot_prod to size of max_time !!!!!!!!!!!!!!!!!!!!
# softmax layer - must make masks
#W_softmax = tf.get_variable(name="bilinear_softmax_weight", shape=[max_time, max_entities], \
#    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
#tf.matmul(dot_prod, W_softmax)
