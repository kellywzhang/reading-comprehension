import tensorflow as tf
import numpy as np
#from testing_gru import rnn
from rnn_cell import GRUCell

state_size = 11
input_size = 3
batch_size = 2
embedding_dim = 3
vocab_size = 100

# Starting interactive Session
sess = tf.InteractiveSession()

inputx = tf.constant([[20,30,40,50,60],[2,3,4,5,0]])

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

for i in range(inputx.get_shape()[1]):
    with tf.variable_scope("Cell{}".format(i)):
        input_ = tf.slice(document_embedding, [0, i, 0], [batch_size, 1, embedding_dim])
        input_ = tf.squeeze(input_)
        output, state = cell(input_, state)
        outputs.append(output)

sess.run(tf.initialize_all_variables())

print(slice1.get_shape())
print(document_embedding.get_shape())
print(input_.get_shape())
print(outputs[0].eval())
print(outputs)

print(document_embedding.eval())
#print(reverse.eval())

sess.close()
