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
