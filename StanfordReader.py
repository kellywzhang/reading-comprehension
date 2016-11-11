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

Issues:
    - Better to pass mask itself instead of repeatedly creating masks with seq_lens?
    - Make softmax numerically stable

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
	def __init__(self, max_entities, hidden_size=128, vocab_size=50000, \
        embedding_dim=100, batch_size=32):

		tf.set_random_seed(1234)

        # Placeholders
        # can add assert statements to ensure shared None dimensions are equal (batch_size)
        seq_lens_d = tf.placeholder(tf.int32, [None, ], name="seq_lens_d")
        seq_lens_q = tf.placeholder(tf.int32, [None, ], name="seq_lens_q")
        input_d = tf.placeholder(tf.int32, [None, None], name="input_d")
        input_q = tf.placeholder(tf.int32, [None, None], name="input_q")
        input_a = tf.placeholder(tf.int32, [None, ], name="input_a")
        input_m = tf.placeholder(tf.int32, [None, ], name="input_m")
        n_steps = tf.placeholder(tf.int32)

        mask_d = tf.cast(tf.sequence_mask(seq_lens_d), tf.int32)
        mask_q = tf.cast(tf.sequence_mask(seq_lens_q), tf.int32)

        # Document and Query embeddings; One-hot-encoded answers
        masked_d = tf.mul(input_d, mask_d)
        masked_q = tf.mul(input_q, mask_q)
        one_hot_a = tf.one_hot(input_a, max_entities)

        # Buildling Graph (Network Layers)
		# ==================================================
        with tf.variable_scope("embedding"):
            W_embeddings = tf.get_variable(shape=[vocab_size, embedding_dim], \
                                           initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                           name="W_embeddings")
            ################## Make option to use pre-trained embeddings ##################

            # Dimensions: batch x max_length x embedding_dim
            document_embedding = tf.gather(W_embeddings, masked_d)
            question_embedding = tf.gather(W_embeddings, masked_q)

        with tf.variable_scope("bidirection_rnn"):
            # Bidirectional RNNs for Document and Question
            forward_cell_d = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="GRU-Forward-D")
            backward_cell_d = GRUCell(state_size=hidden_size,, input_size=embedding_dim, scope="GRU-Backward-D")

            forward_cell_q = GRUCell(state_size=hidden_size,, input_size=embedding_dim, scope="GRU-Forward-Q")
            backward_cell_q = GRUCell(state_size=hidden_size,, input_size=embedding_dim, scope="GRU-Backward-Q")

            hidden_states_d, last_state_d = bidirectional_rnn(forward_cell_d, backward_cell_d, \
                document_embedding, seq_lens_d, batch_size, embedding_dim, concatenate=True)

            hidden_states_q, last_state_q = bidirectional_rnn(forward_cell_q, backward_cell_q, \
                question_embedding, seq_lens_q, batch_size, embedding_dim, concatenate=True)

        with tf.variable_scope("attention"):
            # Attention Layer
            attention = BilinearFunction(attending_size=hidden_size*2, attended_size=hidden_size*2)
            alpha_weights, attend_result = attention(attending=last_state_q, attended=hidden_states_d, \
                seq_lens=seq_lens_d, batch_size=batch_size)

        with tf.variable_scope("prediction"):
            W_predict = tf.get_variable(name="predict_weight", shape=[hidden_size*2, max_entities], \
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            b_predict = tf.get_variable(name="predict_bias", shape=[max_entities],
                initializer=initializer=tf.constant_initializer(0.0))
            # Dimensions (batch_size x state_size/hidden_size*2)
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
            # Negative log-likelihood loss
            self.loss = tf.mul(tf.reduce_sum(log_likelihoods), -1)
            correct_vector = tf.cast(tf.equal(tf.argmax(one_hot_a, 1), tf.argmax(prediction_probs, 1)), \
                tf.float32, name="correct_vector")
            self.accuracy = tf.reduce_mean(correct_vector)
