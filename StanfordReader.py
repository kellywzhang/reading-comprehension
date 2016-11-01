"""
Goal:
    - Takes in batches of (document, question, answer) tuples,
        runs bidirectional rnn, finds attention weights, and calculates loss

Architecture Overview:
    - Bidirectional LSTM/GRU on documents and questions (concatenate depth-wise)
    - Take last outputs of questions (from each direction) as query vector
    - Use bilinear weight to calculate similarity metric/attention weight for
         each word in the document using the query vector
    - Take weighted sum of word vectors and use that to make prediction

Credits: Attentive Reader model developed by https://arxiv.org/pdf/1506.03340.pdf
    and Stanford Reader model developed by https://arxiv.org/pdf/1606.02858v2.pdf
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell

def getFLAGS():
	# Model Hyperparameters
	tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
	tf.flags.DEFINE_integer("num_nodes", 16, "Number of nodes in fully connected layer")
	tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
	tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "Weight lambda on l2 regularization")

	# Training Parameters
	tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
	tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
	tf.flags.DEFINE_integer("patience", 800, "Minimum number of batches seen before early stopping")
	tf.flags.DEFINE_integer("patience_increase", 6, "Number of dev evaluations of increasing loss before early stopping")

	# Display/Saving Parameters
	tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
	tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

	# Print
	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()
	print("\nParameters:")
	for attr, value in sorted(FLAGS.__flags.items()):
	    print("{}={}".format(attr.upper(), value))
	print("")

	return FLAGS

class StanfordReader(object):
    """
    Purpose:
    Instances of this class run the whole StanfordReader model.
    """
	def __init__(self, max_entities, vocab_size=50000, embedding_dim=100, l2_reg_lambda=0):
		tf.set_random_seed(1234)

		# Batch Inputs (documents, questions, answers, entity_mask)
		self.input_d = tf.placeholder(tf.float32, name="input_d")
		self.input_q = tf.placeholder(tf.float32, name="input_q")
        # REFORMAT ANSWER AS ONE HOT VECTOR?
        self.input_a = tf.placeholder(tf.float32, [None, 1], name="input_a")
        self.input_m = tf.placeholder(tf.float32, [None, max_entities], name="input_m")

		# Keeping track of l2 regularization loss
		l2_loss = tf.constant(0.0)

		# Buildling Graph (Network Layers)
		# ==================================================
        W_embeddings = tf.Variable(
			# vocab size, embeding dim, min, max
			tf.random_uniform([vocab_size, embedding_dim], -0.01, 0.01),
			name="W_embeddings")

		document_embedding = tf.gather(W_embeddings, self.input_d)
        question_embedding = tf.gather(W_embeddings, self.input_q)
        #answer_embedding = tf.gather(W_embeddings, self.input_a)

        # Bidirectional RNN
        rnn_cell.GRUCell(num_units=128)

		W = weight_variable(shape=[input_dim, output_dim], name="softmax_weight")
		b = bias_variable(shape=[output_dim], name="softmax_bias")

		y_hat = tf.nn.softmax(tf.matmul(self.input_x, W) + b)

		# Softmax Cross-Entropy Loss
		with tf.name_scope("output"):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_hat, self.input_y)
			self.loss = tf.reduce_mean(cross_entropy) + l2_reg_lambda * l2_loss
			correct_vector = tf.cast(tf.equal(tf.argmax(y_hat, 1), tf.argmax(self.input_y, 1)), tf.float32, name="correct_vector")
			self.accuracy = tf.reduce_mean(correct_vector)

# Helper Functions
# ==================================================

"""
Ideally, I could simply choose the type of layers I want and then have
specific functions for each type of layer and then simply chain the functions together
to form a network

functions for different types of layers (see https://github.com/aymericdamien/TensorFlow-Examples)
"""

def weight_variable(shape, name, initializer="truncated_normal"):
	if initializer == "xavier":
		return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(seed=11))
	return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1, seed=10))

def bias_variable(shape, name):
	return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(value=0.0))
