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
    - Gradient Clipping in GRU

Credits: Attentive Reader model developed by https://arxiv.org/pdf/1506.03340.pdf
    and Stanford Reader model developed by https://arxiv.org/pdf/1606.02858v2.pdf
"""

import tensorflow as tf
import numpy as np
from rnn_cell import GRUCell
from rnn import bidirectional_rnn, rnn
from attention import BilinearFunction

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
    def __init__(self, max_entities, hidden_size=128, vocab_size=50000, embedding_dim=100, batch_size=32):
        self.max_entities = max_entities
        tf.set_random_seed(1234)

        # Placeholders
        # can add assert statements to ensure shared None dimensions are equal (batch_size)
        self.input_d = tf.placeholder(tf.int32, [None, None], name="input_d")
        self.input_q = tf.placeholder(tf.int32, [None, None], name="input_q")
        self.input_a = tf.placeholder(tf.int32, [None, ], name="input_a")
        self.input_m = tf.placeholder(tf.int32, [None, ], name="input_m")

        seq_lens_d = tf.reduce_sum(tf.cast(self.input_d >= 0, tf.int32), 1)
        seq_lens_q = tf.reduce_sum(tf.cast(self.input_q >= 0, tf.int32), 1)

        mask_d = tf.cast(tf.sequence_mask(seq_lens_d), tf.int32)
        mask_q = tf.cast(tf.sequence_mask(seq_lens_q), tf.int32)
        mask_m = tf.cast(tf.sequence_mask(self.input_m, maxlen=max_entities), dtype=tf.float32)

        # Document and Query embddings; One-hot-encoded answers
        masked_d = tf.mul(self.input_d, mask_d)
        masked_q = tf.mul(self.input_q, mask_q)
        one_hot_a = tf.one_hot(self.input_a, self.max_entities)

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
            mask_d = tf.cast(tf.sequence_mask(seq_lens_d), tf.float32) #or float64?
            mask_q = tf.cast(tf.sequence_mask(seq_lens_q), tf.float32)

            # Bidirectional RNNs for Document and Question
            self.forward_cell_d = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="GRU-Forward-D")
            self.backward_cell_d = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="GRU-Backward-D")

            self.forward_cell_q = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="GRU-Forward-Q")
            self.backward_cell_q = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="GRU-Backward-Q")

            self.hidden_states_d, last_state_d = bidirectional_rnn(self.forward_cell_d, self.backward_cell_d, \
                document_embedding, mask_d, concatenate=True)

            hidden_states_q, self.last_state_q = bidirectional_rnn(self.forward_cell_q, self.backward_cell_q, \
                question_embedding, mask_q, concatenate=True)

        with tf.variable_scope("attention"):
            # Attention Layer
            self.attention = BilinearFunction(attending_size=hidden_size*2, attended_size=hidden_size*2)
            self.alpha_weights, self.attend_result = self.attention(attending=self.last_state_q, attended=self.hidden_states_d, \
                time_mask=mask_d)

        with tf.variable_scope("prediction"):
            self.W_predict = tf.get_variable(name="predict_weight", shape=[hidden_size*2, self.max_entities], \
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            self.b_predict = tf.get_variable(name="predict_bias", shape=[self.max_entities],
                initializer=tf.constant_initializer(0.0))
            # Dimensions (batch_size x max_entities)
            self.predict_probs = (tf.matmul(self.attend_result, self.W_predict) + self.b_predict) * mask_m

            # Custom Softmax b/c need to use time_mask --------------------
            # Also numerical stability:

            # e_x = exp(x - x.max(axis=1))
            # out = e_x / e_x.sum(axis=1)
            numerator = tf.exp(tf.sub(self.predict_probs, tf.expand_dims(tf.reduce_max(self.predict_probs, 1), -1))) * mask_m
            denom = tf.reduce_sum(numerator, 1)

            # Transpose so broadcasting scalar division works properly
            # Dimensions (batch x max_entities)
            #self.predict_probs_normalized = tf.transpose(tf.div(tf.transpose(numerator), denom))
            self.predict_probs_normalized = tf.div(numerator, tf.expand_dims(denom, 1))
            self.likelihoods = tf.reduce_sum(tf.mul(self.predict_probs_normalized, one_hot_a), 1)
            log_likelihoods = tf.log(self.likelihoods+0.00000000000000000001)
            # Negative log-likelihood loss
            self.loss = tf.mul(tf.reduce_sum(log_likelihoods), -1)
            correct_vector = tf.cast(tf.equal(tf.argmax(one_hot_a, 1), tf.argmax(self.predict_probs_normalized, 1)), \
                tf.float32, name="correct_vector")
            self.accuracy = tf.reduce_mean(correct_vector)


    def get_mask_shape(self):
           print (self.mask_d.get_shape(), self.mask_q.get_shape())
