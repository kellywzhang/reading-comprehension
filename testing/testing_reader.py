import tensorflow as tf
import numpy as np
import os
import sys
os.path.abspath(os.path.curdir)[:-8]
sys.path.insert(0, os.path.abspath(os.path.curdir)[:-8])

from rnn_cell import GRUCell
from rnn import rnn, bidirectional_rnn

# Parameters
max_entities = 10
hidden_size = 128
vocab_size = 50000
embedding_dim = 8
batch_size = 2
state_size = 11
input_size = 8

# Starting interactive Session
sess = tf.InteractiveSession()

# Placeholders
# can add assert statements to ensure shared None dimensions are equal (batch_size)
seq_lens = tf.placeholder(tf.int32, [None, ], name="seq_lens")
input_d = tf.placeholder(tf.int32, [None, None], name="input_d")
input_q = tf.placeholder(tf.int32, [None, None], name="input_q")
input_a = tf.placeholder(tf.int32, [None, ], name="input_a")
input_m = tf.placeholder(tf.int32, [None, ], name="input_m")
n_steps = tf.placeholder(tf.int32)

# toy feed dict
feed = {
    n_steps: 5,
    seq_lens: [5,4],
    input_d: [[20,30,40,50,60],[2,3,4,5,-1]], # document
    input_q: [[2,3,-1],[1,2,3]],              # query
    input_a: [40,5],                          # answer
    input_m: [2,3],                           # number of entities
}

# create 0, 1 masks (not boolean b/c tf's boolean_mask function reduces dimension of output)
mask_d = tf.cast(input_d >= 0, tf.int32)
mask_q = tf.cast(input_q >= 0, tf.int32)

masked_d = tf.mul(input_d, mask_d)
masked_q = tf.mul(input_q, mask_q)

with tf.variable_scope("embedding"):
    W_embeddings = tf.get_variable(shape=[vocab_size, embedding_dim], \
                                   initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                   name="W_embeddings")

    # Dimensions: batch x max_length x embedding_dim
    document_embedding = tf.gather(W_embeddings, masked_d)
    question_embedding = tf.gather(W_embeddings, masked_q)

    timesteps_d = document_embedding.get_shape().as_list()[1]
    #print(timesteps_d)
    timesteps_q = question_embedding.get_shape()[1]

    #tf.split(1, timesteps_d, document_embedding)

    slice1 = tf.slice(document_embedding, [0, 0, 0], [batch_size, 1, embedding_dim])
    slice2 = tf.slice(document_embedding, [0, 1, 0], [batch_size, 2, embedding_dim])
    #for i in range(batch_size):
    #    slicei = tf.squeeze(slice1[i])

    cell = GRUCell(state_size, input_size)
    state = cell.zero_state(batch_size)
    outputs, last_state = rnn(cell, document_embedding, seq_lens, batch_size, embedding_dim)
    #output = cell(slice1[0], state)

# Attention testing
big_tensor = tf.constant([[[1,2,3,4],[4,5,6,7]], [[1,2,3,4],[4,5,6,7]], [[1,2,3,4],[4,5,6,7]]])
flat = tf.reshape(big_tensor, [-1, 4])
tensor = tf.constant([[1.0,2.,3.,4.],[4.,5.,6.,7.]])
vector = tf.constant(np.transpose([[1.,1.,1.,1.]]), dtype=tf.float32)

prod = tf.matmul(tensor, vector)

maskk = tf.sequence_mask([1,2,3])



cell = GRUCell(state_size, input_size)
state = cell.zero_state(batch_size)

seq_len_mask = tf.cast(tf.sequence_mask(seq_lens), tf.float32)
time = tf.reduce_max(seq_lens)

def condition(i, inputs, state, outputs):
    return tf.less(i, time)

def body(i, inputs, state, outputs):
    with tf.variable_scope("Cell{}".format(1)):
        input_ = tf.slice(inputs, [0, i, 0], [batch_size, 1, embedding_dim])
        time_mask = tf.slice(seq_len_mask, [0, i], [batch_size, 1])
        input_ = tf.squeeze(input_)
        output, state = cell(input_, state, time_mask)
        outputs = tf.concat(1, [outputs, output])
    return [tf.add(i, 1), inputs, state, outputs]

i = tf.constant(0)
_, _, last_states, hidden_states = tf.while_loop(condition, body, \
    loop_vars=[i, document_embedding, state, tf.zeros([batch_size, 1])], \
    shape_invariants=[i.get_shape(), document_embedding.get_shape(), state.get_shape(), tf.TensorShape([batch_size, None])])
# get rid of zero output start state for concat purposes
hidden_states = tf.slice(hidden_states, [0, 1], [batch_size, -1])
hidden_states = tf.reshape(hidden_states, [batch_size, -1, state_size])

# for softmax: tf.sequence_mask
#document_embedding
reverse = tf.reverse_sequence(document_embedding, seq_lens, seq_dim=1, batch_dim=0)

inputs = document_embedding
reverse_inputs = tf.reverse_sequence(inputs, seq_lens, seq_dim=1, batch_dim=0)

forward_cell = GRUCell(state_size, input_size, scope="GRU-Forward")
backward_cell = GRUCell(state_size, input_size, scope="GRU-Backward")

#forward_outputs, forward_last_state = rnn(forward_cell, inputs, seq_lens, batch_size, embedding_dim)
#backward_outputs, backward_last_state = rnn(backward_cell, reverse_inputs, seq_lens, batch_size, embedding_dim)

#LS = tf.concat(1, [forward_last_state, backward_last_state])
#LSS = tf.concat(2, [forward_outputs, backward_outputs])

LS, LSS = bidirectional_rnn(forward_cell, backward_cell, inputs, seq_lens, batch_size, embedding_dim, concatenate=True)

sess.run(tf.initialize_all_variables())
"""
print(forward_last_state.eval(feed))
print(backward_last_state.eval(feed))
print(forward_last_state.eval(feed).shape) # batch x hidden_state
"""
print(LS.eval(feed))
print(LS.eval(feed).shape)
print(LSS.eval(feed))
print(LSS.eval(feed).shape)

#print(forward_outputs.eval(feed))
#print(backward_outputs.eval(feed))
#print(backward_outputs.eval(feed).shape)

"""print(outputs[0].eval(feed))
print(outputs[1].eval(feed))
print(outputs[2].eval(feed))
print(outputs[3].eval(feed))
print(outputs[4].eval(feed))
print(state.eval(feed))
print(n_steps.eval(feed))"""
#print(time.eval(feed))
#print(last_state.eval(feed))
#print(hidden_states.eval(feed))
"""print(r[0].eval(feed))
print(r[1].eval(feed))
print(r[2].eval(feed))
print(r[3].eval(feed))"""
#print(ijk_final.eval())

#print(outputs.eval(feed))
#print(last_state.eval(feed))
#print(document_embedding.eval(feed))
#print(reverse.eval(feed))

"""
# Attention testing
print(maskk.eval())
print(tensor.get_shape())
print(vector.get_shape())
print(flat.get_shape())
print(flat.eval())
print(prod.eval())
"""
"""print(input_m.get_shape())
print(input_d.eval(feed))
print(mask_d.eval(feed))
print(masked_d.eval(feed))
print(input_d.get_shape())
print(masked_d.get_shape())


print(input_d.eval(feed).shape)"""
"""
slices = slice1.eval(feed)
print(type(slices))
print(slices.shape)

print(slice1[0].eval(feed))
print(slice1[1].eval(feed))

#print(slicei.eval(feed).shape)

print(slice1.eval(feed))
print(slice2.eval(feed))
print(state.get_shape())

print(document_embedding.eval(feed).shape)
"""
sess.close()
