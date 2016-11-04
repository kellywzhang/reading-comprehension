"""
Goal:
    - Create RNN layers

TODO/ISSUES:

Credits: Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
"""

import TensorFlow as tf

def rnn(cell, inputs):
    state = cell.zero_state()
    outputs = []

    # FIX THIS LOOP
    tf.slice(input_, [0, i, 0], [batch_size, 1, embedding_dim])

    for input_ in inputs:
      output, state = cell(input_, state)
      outputs.append(output)

    return (outputs, state)

def bidirectional_rnn(forward_cell, backward_cell, inputs, concatenate=True):
    forward_outputs, forward_last_state = rnn(forward_cell, inputs)
    backward_outputs, backward_last_state = rnn(backward_cell, inputs)

    if concatenate:
        # FIGURE OUT PROPER DIMENSIONS
        #outputs = tf.concatenate(axis, [foward_outputs, backward_outputs])
        #last_state = tf.concatenate(axis, [forward_last_state, backward_last_state])
        return (outputs, last_state)

    return (forward_outputs, forward_last_state, backward_outputs, backward_last_state)
