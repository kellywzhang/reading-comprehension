import tensorflow as tf
import numpy as np
import sys
os.path.abspath(os.path.curdir)[:-8]
sys.path.insert(0, os.path.abspath(os.path.curdir)[:-8])

from rnn_cell import GRUCell
from rnn import rnn

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

"""def condition(x, y):
    return tf.less(x, time)

def body(x, y):
    return tf.add(x, y)

time = tf.constant(5)
i = tf.constant(0)
r = tf.while_loop(condition, body, [i, time])"""

a = tf.ones([1000])
b = tf.ones([1000])

cond = lambda i, a, b: tf.less(i, int(1e6))
body = lambda i, a, b: [tf.add(i, 1), a * b, b]

i = tf.constant(0)
output = tf.while_loop(cond, body, [i, a, b])

# for softmax: tf.sequence_mask

sess.run(tf.initialize_all_variables())
"""print(outputs[0].eval(feed))
print(outputs[1].eval(feed))
print(outputs[2].eval(feed))
print(outputs[3].eval(feed))
print(outputs[4].eval(feed))
print(state.eval(feed))
print(n_steps.eval(feed))"""
#print(time.eval(feed))
#print(r.eval(feed))
print(output)
print(output[0].eval())
