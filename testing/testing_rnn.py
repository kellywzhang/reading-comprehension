import tensorflow as tf
import numpy as np
import sys
import os
os.path.abspath(os.path.curdir)[:-8]
sys.path.insert(0, os.path.abspath(os.path.curdir)[:-8])

from rnn_cell import GRUCell

state_size = 11
input_size = 3
batch_size = 2
embedding_dim = 3
vocab_size = 100

# Starting interactive Session
sess = tf.InteractiveSession()

inputx = tf.constant([[20,30,40,50,60],[2,3,4,5,0]])
seq_lens = tf.constant([5,4])

W_embeddings = tf.get_variable(shape=[vocab_size, embedding_dim], \
                               initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                               name="W_embeddings")

# Dimensions: batch x max_length x embedding_dim
document_embedding = tf.gather(W_embeddings, inputx)

slice1 = tf.slice(document_embedding, [0, 2, 0], [batch_size, 1, embedding_dim])

#reverse = tf.reverse_sequence(document_embedding, [5, 5], 1)

cell = GRUCell(state_size, input_size)
state = cell.zero_state(batch_size)

# RNN
outputs = []
print(dir(tf.reduce_max(seq_lens)))
print(tf.reduce_max(seq_lens).__str__)

seq_len_mask = tf.cast(tf.sequence_mask(seq_lens), tf.float32)
anti_time_mask = tf.cast(seq_len_mask<=0, tf.float32)

#for i in range(inputx.get_shape()[1]):

a = tf.reduce_max(seq_lens).eval()

for i in range(tf.reduce_max(seq_lens).eval()):
    with tf.variable_scope("Cell{}".format(i)):
        input_ = tf.slice(document_embedding, [0, i, 0], [batch_size, 1, embedding_dim])
        input_ = tf.squeeze(input_)
        time_mask = tf.slice(seq_len_mask, [0, i], [batch_size, 1])
        output, state = cell(input_, state, time_mask)
        outputs.append(output)

"""def a(x):
    return x
 = tf.scan(a, seq_lens)"""

elems = tf.transpose(inputx) #np.array([1, 2, 3, 4, 5, 6])

seqimp = tf.scan(lambda a, x: a + x, elems)

c = tf.constant(3)
l = tf.constant([20,30,40,50,60,2,3,4,5,0])


temp = tf.placeholder(tf.int32, [None])
feed = {temp:[20,30,40,50,60,2,3,4,5,0]}
#assert l.get_shape() == temp.get_shape()

aaa = tf.constant([x for x in range(tf.shape(temp)[0])])

tf.assert_equal(tf.shape(temp), 10)#tf.shape(l))
ab = l >= c

sess.run(tf.initialize_all_variables())

print(slice1.get_shape())
print(document_embedding.get_shape())
print(input_.get_shape())
print(outputs[0].eval())
print(outputs[1].eval())
print(outputs[2].eval())
print(outputs[3].eval())
print(outputs[4].eval())
print(outputs)

print(a)

print(seq_len_mask.eval())
print(anti_time_mask.eval())

print(seqimp.eval())
print(ab.eval())
print(temp.eval(feed))
print(aaa.eval(feed))

#print(document_embedding.eval())
#print(reverse.eval())

sess.close()
