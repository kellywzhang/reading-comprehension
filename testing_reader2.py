import tensorflow as tf
import numpy as np
from rnn_cell import GRUCell
from rnn import rnn, bidirectional_rnn
from attention import BilinearFunction

"""
Gradient clipping
passing mask instead of seq_lens
"""


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
seq_lens_d = tf.placeholder(tf.int32, [None, ], name="seq_lens_d")
seq_lens_q = tf.placeholder(tf.int32, [None, ], name="seq_lens_q")
input_d = tf.placeholder(tf.int32, [None, None], name="input_d")
input_q = tf.placeholder(tf.int32, [None, None], name="input_q")
input_a = tf.placeholder(tf.int32, [None, ], name="input_a")
input_m = tf.placeholder(tf.int32, [None, ], name="input_m")
n_steps = tf.placeholder(tf.int32)

# toy feed dict
feed = {
    n_steps: 5,
    seq_lens_d: [5,4],
    seq_lens_q: [2,3],
    input_d: [[20,30,40,50,60],[2,3,4,5,0]], # document
    input_q: [[2,3,0],[1,2,3]],              # query
    input_a: [1,0],                          # answer
    input_m: [2,3],                           # number of entities
}

mask_d = tf.cast(tf.sequence_mask(seq_lens_d), tf.int32)
mask_q = tf.cast(tf.sequence_mask(seq_lens_q), tf.int32)

masked_d = tf.mul(input_d, mask_d)
masked_q = tf.mul(input_q, mask_q)

one_hot_a = tf.one_hot(input_a, max_entities)

"""
Better to pass mask itself instead of repeatedly creating masks with seq_lens?
"""

with tf.variable_scope("embedding"):
    W_embeddings = tf.get_variable(shape=[vocab_size, embedding_dim], \
                                   initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                   name="W_embeddings")

    # Dimensions: batch x max_length x embedding_dim
    document_embedding = tf.gather(W_embeddings, masked_d)
    question_embedding = tf.gather(W_embeddings, masked_q)

with tf.variable_scope("bidirection_rnn"):
    # Bidirectional RNNs for Document and Question
    forward_cell_d = GRUCell(state_size, input_size, scope="GRU-Forward-D")
    backward_cell_d = GRUCell(state_size, input_size, scope="GRU-Backward-D")

    forward_cell_q = GRUCell(state_size, input_size, scope="GRU-Forward-Q")
    backward_cell_q = GRUCell(state_size, input_size, scope="GRU-Backward-Q")

    hidden_states_d, last_state_d = bidirectional_rnn(forward_cell_d, backward_cell_d, \
        document_embedding, seq_lens_d, batch_size, embedding_dim, concatenate=True)

    hidden_states_q, last_state_q = bidirectional_rnn(forward_cell_q, backward_cell_q, \
        question_embedding, seq_lens_q, batch_size, embedding_dim, concatenate=True)

with tf.variable_scope("attention"):
    # Attention Layer
    attention = BilinearFunction(attending_size=state_size*2, attended_size=state_size*2)
    alpha_weights, attend_result = attention(attending=last_state_q, attended=hidden_states_d, \
        seq_lens=seq_lens_d, batch_size=batch_size)

with tf.variable_scope("prediction"):
    W_predict = tf.get_variable(name="predict_weight", shape=[state_size*2, max_entities], \
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
    b_predict = tf.get_variable(name="predict_bias", shape=[max_entities],
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
    # Dimensions (batch_size x state_size*2)
    prediction_probs_unnormalized = tf.matmul(attend_result, W_predict) + b_predict

    # Custom Softmax b/c need to use time_mask --------------------
    # Also numerical stability:
    mask_m = tf.cast(tf.sequence_mask(input_m, maxlen=10), tf.float32)
    numerator = tf.exp(prediction_probs_unnormalized) * mask_m
    denom = tf.reduce_sum(tf.exp(prediction_probs_unnormalized) * mask_m, 1)

    # Transpose so broadcasting scalar division works properly
    # Dimensions (batch x time)
    prediction_probs = tf.transpose(tf.div(tf.transpose(numerator), denom))
    likelihoods = tf.reduce_sum(tf.mul(prediction_probs, one_hot_a), 1)
    log_likelihoods = tf.log(likelihoods)
    loss = tf.mul(tf.reduce_sum(log_likelihoods), -1)
    correct_vector = tf.cast(tf.equal(tf.argmax(one_hot_a, 1), tf.argmax(prediction_probs, 1)), \
        tf.float32, name="correct_vector")
    accuracy = tf.reduce_mean(correct_vector)

sess.run(tf.initialize_all_variables())

print(alpha_weights.eval(feed))
print(attend_result.eval(feed))
print(attend_result.get_shape())
print(mask_m.eval(feed))
print(numerator.get_shape())
print(prediction_probs.eval(feed))
print(one_hot_a.eval(feed))
print(likelihoods.eval(feed))
print(log_likelihoods.eval(feed))
print(loss.eval(feed))
print(correct_vector.eval(feed))
print(accuracy.eval(feed))

sess.close()
