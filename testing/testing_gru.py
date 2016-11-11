import tensorflow as tf
import numpy as np
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid

state_size = 10
input_size = 8
batch_size = 2

# Starting interactive Session
sess = tf.InteractiveSession()

state = tf.zeros([batch_size, state_size])
word = tf.placeholder(tf.float32, [batch_size, input_size], name="input_d")
inputs = tf.concat(1, [state, word])

with tf.variable_scope("Gates"):  # Reset gate and update gate.
  # We start with bias of 1.0 to not reset and not update.
  W_reset = tf.get_variable(name="reset_weight", shape=[state_size+input_size, state_size], \
      initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
  W_update = tf.get_variable(name="update_weight", shape=[state_size+input_size, state_size], \
      initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
  b_reset = tf.get_variable(name="reset_bias", shape=[state_size], initializer=tf.constant_initializer(0.0))
  b_update = tf.get_variable(name="update_bias", shape=[state_size], initializer=tf.constant_initializer(0.0))

  reset = sigmoid(tf.matmul(inputs, W_reset) + b_reset)
  update = sigmoid(tf.matmul(inputs, W_update) + b_update)


with tf.variable_scope("Candidate"):
  W_candidate = tf.get_variable(name="candidate_weight", shape=[state_size+input_size, state_size], \
      initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
  b_candidate = tf.get_variable(name="candidate_bias", shape=[state_size], \
      initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

  reset_input = tf.concat(1, [reset * state, word])
  candidate = tanh(tf.matmul(reset_input, W_reset) + b_candidate)

new_h = update * state + (1 - update) * candidate

### WORKS!!!


sess.run(tf.initialize_all_variables())

feed_dict = {word: [[1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8]]}
print(state.get_shape())
print(word.get_shape())
print("hi")
print(inputs.eval(feed_dict))
print(reset.eval(feed_dict))
print(inputs.eval(feed_dict).shape)
print(W_reset.get_shape())

print(candidate.eval(feed_dict))
print(new_h.eval(feed_dict))
print(new_h.eval(feed_dict).shape)

sess.close()

def zero_state():
  return tf.Variable(tf.zeros([state_size]))
