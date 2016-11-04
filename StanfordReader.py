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
import rnn_cell, rnn

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
	def __init__(self, max_entities, hidden_size=128, vocab_size=50000, embedding_dim=100):
		tf.set_random_seed(1234)

		# Batch Inputs (documents, questions, answers, entity_mask)
        # Dimensions: batch x max_length
		self.input_d = tf.placeholder(tf.float32, name="input_d")
		self.input_q = tf.placeholder(tf.float32, name="input_q")
        self.input_a = tf.placeholder(tf.float32, [None, 1], name="input_a")
        self.input_m = tf.placeholder(tf.float32, [None, max_entities], name="input_m")

		# Buildling Graph (Network Layers)
		# ==================================================
        W_embeddings = tf.get_variable(shape=[vocab_size, embedding_dim], \
			initializer=tf.random_uniform_initializer(-0.01, 0.01),
			name="W_embeddings")
        ################## Make option to use pre-trained embeddings ##################

        embeddings_with_pad = tf.concat(0, [W_embeddings, tf.zeros(shape=[1, embedding_dim])])

        # Dimensions: batch x max_length x embedding_dim
		document_embedding = tf.gather(embeddings_with_pad, self.input_d)
        question_embedding = tf.gather(embeddings_with_pad, self.input_q)
        #answer_embedding = tf.gather(embeddings_with_pad, self.input_a)

        # Bidirectional RNN (for both Document and Query)
        ### Document
        num_steps_d = self.input_d.get_shape()[1]
        gru_cell_d_forward = rnn_cell.GRUCell(state_size=128, input_size=embedding_dim, scope="gru_document_forward")
        gru_cell_d_backward = rnn_cell.GRUCell(state_size=128, input_size=embedding_dim, scope="gru_document_backward")
        # Dimension: batch x time x hidden_size*2; batch x 1 x hidden_size*2
        outputs_d, last_state_d = rnn.bidirectional_rnn(gru_cell_d_forward, gru_cell_d_backward, self.input_d, concatenate=True)
        # make mask for documents before attention layer

        ### Query
        num_steps_q = self.input_q.get_shape()[1]
        gru_cell_q_forward = rnn_cell.GRUCell(state_size=128, input_size=embedding_dim, scope="gru_query_forward")
        gru_cell_q_backward = rnn_cell.GRUCell(state_size=128, input_size=embedding_dim, scope="gru_query_backward")
        outputs_q, last_state_q = rnn.bidirectional_rnn(gru_cell_q_forward, gru_cell_q_backward self.input_d, concatenate=False)
        # must get correct timesteps for outputs!!!!!!!!

        # Attention Layer
        W_attention = tf.get_variable(name="attention_weight", shape=[hidden_size*2, hidden_size*2], \
            initializer=tf.random_uniform_initializer(-0.01, 0.01))
        attention_metric = matmul(query_vector, W_attention)
        attention_weights = matmul(attention_metric, outputs_d) # do these dimensions work???, need batch matmul?

        # scalar multiply output_d by attention weights, then tf.gather

		W_softmax = tf.get_variable(shape=[hidden_size*2, max_num_entities], name="softmax_weight", \
            initializer==tf.random_uniform_initializer(-0.01, 0.01))
		b_softmax = tf.get_variable(shape=[max_num_entities]], name="softmax_bias", \
            initializer=initializer=tf.constant_initializer(0.0))

		y_hat = tf.nn.softmax(tf.matmul(attention-weighted-document, W_softmax) + b_softmax)

		# Softmax Cross-Entropy Loss
		with tf.name_scope("output"):
            # HAVE TO MAKE self.input_a one-hot
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_hat, self.input_a)
            # MASK HERE?? How to perform softmax over different number of inputs within batch?
			self.loss = tf.reduce_mean(cross_entropy)
			correct_vector = tf.cast(tf.equal(tf.argmax(y_hat, 1), tf.argmax(self.input_a, 1)), tf.float32, name="correct_vector")
			self.accuracy = tf.reduce_mean(correct_vector)
