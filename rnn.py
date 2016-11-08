"""
Goal:
    - Create RNN layers

TODO/ISSUES:

Credits: Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
"""

import TensorFlow as tf

def rnn(cell, inputs, seq_len):
    state = cell.zero_state()
    outputs = []

    #seq_len_mask = tf.cast(tf.sequence_mask(seq_len), tf.int32)

    #tf.reduce_max(seq_lens)
    for i in range(seq_len):
        with tf.variable_scope("Cell{}".format(i)):
            input_ = tf.slice(document_embedding, [0, i, 0], [batch_size, 1, embedding_dim])
            #time_mask = tf.slice(seq_len_mask, [0, i], [batch_size, 1])
            input_ = tf.squeeze(input_)
            output, state = cell(input_, state)
            #output, state = cell(input_, state, time_mask)
            outputs.append(output)

    return (outputs, state)

def bidirectional_rnn(forward_cell, backward_cell, inputs, concatenate=True):
    #Reverse inputs using tf.reverse_sequence(input, seq_lengths, seq_dim, batch_dim=None, name=None)
    # Add seqLen params

    forward_outputs, forward_last_state = rnn(forward_cell, inputs)
    backward_outputs, backward_last_state = rnn(backward_cell, inputs)

    if concatenate:
        # FIGURE OUT PROPER DIMENSIONS
        #outputs = tf.concatenate(axis, [foward_outputs, backward_outputs])
        #last_state = tf.concatenate(axis, [forward_last_state, backward_last_state])
        return (outputs, last_state)

    return (forward_outputs, forward_last_state, backward_outputs, backward_last_state)
