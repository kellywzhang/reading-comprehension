import tensorflow as tf
import numpy as np
from rnn_cell import GRUCell
from rnn import bidirectional_rnn, rnn
from attention import BilinearFunction

# sentence delimiting punctuation: . ; ! ?
# assume data is tokenized
# use vocabulary dictionary to find the indices of these special characters

"""
Ideas for Improvements
======================
- RNN for query representation
self.forward_cell_q = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="GRU-Forward-Q")
self.backward_cell_q = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="GRU-Backward-Q")

- Bilinear attention instead of cosine similarity

"""


# Parameters
batch_size = 2
embedding_size = 3
vocab_size = 20

# First Session
# Read query with bidirectional RNN
# gather indices for

# Max number of sentences in any document
# Sentence mask (sequences)
# Number of words in each sentence

sess = tf.InteractiveSession()

# (batch_size * max_sentence_count x max_sentence_length)
sentences = tf.placeholder(tf.int32, [None, None], name="sentences")
doc_lengths = tf.placeholder(tf.int32, [batch_size, ], name="doc_lengths")
questions = tf.placeholder(tf.int32, [batch_size, None], name="questions")

labels = tf.placeholder(tf.int32, [batch_size, ], name="labels") #***

feed_dict = {
    sentences: [[3,4,5,6,3,4], [3,4,5,6,3,4], [3,4,5,6,-1,-1], [-1,-1,-1,-1,-1,-1]],
    doc_lengths: [2,1],
    questions: [[1,2,3],[9,2,-1]],
    labels: [1, 0] #***
}

# CREATING SENTENCE REPRESENTATIONS FOR ALL DOCUMENTS ---------------------------
# create mask ALTERNATIVE*************************************

# (batch_size * mask_sentence_count x max_sentence_length)
sentence_mask = tf.cast(sentences >= 0, tf.int32)
#sentence_mask = tf.sequence_mask(doc_lengths, dtype=tf.int32)
masked_sentences = tf.mul(sentences, sentence_mask)

max_sent_length = tf.shape(sentence_mask)[1]
max_sent_per_doc = tf.cast(tf.shape(sentence_mask)[0]/batch_size, tf.int32)


# (batch_size x max_sent_per_doc)
batch_mask = tf.reshape(tf.reduce_max(sentence_mask, 1), [batch_size, -1])
# (batch_size * max_sent_per_doc x 1 x 1)
sentence_batch_mask = tf.cast(tf.reshape(batch_mask, [-1, 1, 1]), tf.float32)


# Grab embeddings
W_embeddings = tf.get_variable(shape=[vocab_size, embedding_size], \
                               initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                               name="W_embeddings")
# batch_size * max_sent_per_doc x max_sentence_length x embedding_size
sentence_embeddings = tf.gather(W_embeddings, masked_sentences)
masked_sentence_embeddings = tf.mul(sentence_embeddings, tf.cast(tf.expand_dims(sentence_mask, -1), tf.float32))

# CBOW
# (batch_size * max_sentence_count x embedding_size)
cbow_sentences = tf.reduce_mean(masked_sentence_embeddings, 1)
# reshape batch to (batch_size x max_doc_length x embedding_size)
doc_sentences = tf.reshape(cbow_sentences, [batch_size, -1, embedding_size])


# CREATING QUESTION REPRESENTATIONS FOR ALL EXAMPLES ----------------------------

# create mask
# (batch_size x max_question_length)
question_mask = tf.cast(questions > 0, tf.int32)
masked_question = tf.mul(question_mask, questions)

# easy baseline: cbow
# (batch_size x max_question_length x embedding_size)
question_embeddings = tf.gather(W_embeddings, masked_question)
question_mask_float = tf.expand_dims(tf.cast(question_mask, tf.float32), -1)
masked_question_embeddings = tf.mul(question_embeddings, question_mask_float)
# (batch_size x embedding_size)
question_cbow = tf.reduce_mean(masked_question_embeddings, 1)

# can use RNN representation as well*************************************

# ATTENTION/SIMILARITY SCORING --------------------------------------------------
# Using simple dot product/cosine similiarity as of now (https://arxiv.org/pdf/1605.07427v1.pdf)

attention = BilinearFunction(attending_size=embedding_size, attended_size=embedding_size)
alpha_weights, attend_result = attention(attending=question_cbow, attended=doc_sentences, \
    time_mask=tf.cast(batch_mask, tf.float32))


# (batch_size x max_sent_per_doc)
dot_prod = tf.squeeze(tf.batch_matmul(doc_sentences, tf.expand_dims(question_cbow, -1)), [-1])

# (batch_size x max_sent_per_doc)
sentence_norm = tf.sqrt(tf.reduce_sum(tf.mul(doc_sentences, doc_sentences), -1))
# (batch_size)
question_norm = tf.sqrt(tf.reduce_sum(tf.mul(question_cbow, question_cbow), 1))

denom = tf.mul(sentence_norm, tf.expand_dims(question_norm, -1))+1e-30
# (batch_size x max_sent_per_doc) - scalars between -1 and +1
cosine_similarity = tf.div(dot_prod, denom)

####
# classification step

masked_pos_cos_sim = tf.sub(tf.add(cosine_similarity, 1), tf.cast(batch_mask < 1, tf.float32))

normalized_cos_sim = tf.div(masked_pos_cos_sim, tf.expand_dims(tf.reduce_sum(masked_pos_cos_sim, 1), -1))
one_hot_labels = tf.one_hot(labels, max_sent_per_doc)

likelihoods = tf.reduce_sum(tf.mul(normalized_cos_sim, one_hot_labels), 1)
log_likelihoods = tf.log(likelihoods+0.00000000000000000001)
loss = tf.mul(tf.reduce_sum(log_likelihoods), -1)
correct_vector = tf.cast(tf.equal(labels, tf.cast(tf.argmax(normalized_cos_sim, 1), tf.int32)), \
    tf.float32, name="correct_vector")
accuracy = tf.reduce_mean(correct_vector)

####

# Must create special version of batch_mask to ensure that that zero values are less than the top_k values
anti_batch_mask = tf.mul(tf.cast(batch_mask < 1, tf.float32), 2)
masked_cosine_similarity = tf.sub(cosine_similarity, anti_batch_mask)

# top_k.indices; top_k.values
top_k_indices = tf.nn.top_k(masked_cosine_similarity, k=1, sorted=True, name="top_k").indices
#top_k_values = tf.nn.top_k(masked_cosine_similarity, k=1, sorted=True, name="top_k").values

# (batch_size x max_doc_length)
top_k_mask = tf.one_hot(top_k_indices, max_sent_per_doc, on_value=1, off_value=0, dtype=tf.int32, name="top_k_mask")
top_k_mask_sentences = tf.squeeze(tf.cast(tf.reshape(top_k_mask, [batch_size*max_sent_per_doc, -1]), tf.bool))

top_k_indices_sentences = tf.where(top_k_mask_sentences)
# (batch_size x k x max_sent_length x embedding_size)
top_k_sentences = tf.gather(masked_sentence_embeddings, top_k_indices_sentences)





sess.run(tf.initialize_all_variables())
print(alpha_weights.eval(feed_dict))
# print(doc_sentences.eval(feed_dict).shape)
# print(accuracy.eval(feed_dict))
# print(one_hot_labels.eval(feed_dict))
# print(likelihoods.eval(feed_dict))
# print(correct_vector.eval(feed_dict))
# print(cosine_similarity.eval(feed_dict))
# print(masked_pos_cos_sim.eval(feed_dict))
# print(normalized_cos_sim.eval(feed_dict))
# # print(question_cbow.eval(feed_dict))
# print(doc_sentences.eval(feed_dict))
# print(masked_sentence_embeddings.eval(feed_dict))
# print(masked_cosine_similarity.eval(feed_dict))
# print(top_k_indices.eval(feed_dict))
# print(top_k_mask.eval(feed_dict))
# print(top_k_mask_sentences.eval(feed_dict))
# print(top_k_indices_sentences.eval(feed_dict))
# print(top_k_sentences.eval(feed_dict))
# print(top_k_sentences.eval(feed_dict).shape)

# print(dot_prod.eval(feed_dict))
# print(sentence_norm.eval(feed_dict).shape)
# print(question_norm.eval(feed_dict).shape)
# print(denom.eval(feed_dict))
# print("hi")
# print(question_cbow.eval(feed_dict))
# print(question_norm.eval(feed_dict))
# print(question_norm.eval(feed_dict).shape)
# print(sentence_norm.eval(feed_dict))
# print(sentence_norm.eval(feed_dict).shape)
# print(denom.eval(feed_dict))
# print("hi")
# print(cosine_similarity.eval(feed_dict))
# print(masked_cosine_similarity.eval(feed_dict))
# print(anti_batch_mask.eval(feed_dict))
# print(doc_sentences.eval(feed_dict))
# print(doc_sentences.eval(feed_dict).shape)

# print("batch_mask")
# print(batch_mask.eval(feed_dict))
# print(batch_mask.eval(feed_dict).shape)
# print(anti_batch_mask.eval(feed_dict))
# print(min_alpha_weight.eval(feed_dict))
# # print("alpha weight mask")
# # print(alpha_weight_mask.eval(feed_dict))
# print(masked_dot_prod.eval(feed_dict))
# print(masked_dot_prod.eval(feed_dict).shape)

# print("masked_sentences")
# print(masked_sentences.eval(feed_dict))
# print(masked_sentences.eval(feed_dict).shape)
# print("top k mask")
# print(top_k_mask_sentences.eval(feed_dict))
# print(top_k_mask_sentences.eval(feed_dict).shape)
# # print("top k sentences")
# # print(top_k_sentences.eval(feed_dict))
# # print(top_k_sentences.eval(feed_dict).shape)
# print("alpha weights")
# print(masked_dot_prod.eval(feed_dict))
# print(dot_prod.eval(feed_dict).shape)
# print("top k indices")
# print(top_k_indices.eval(feed_dict))
# print(top_k_indices.eval(feed_dict).shape)
# print("top k mask")
# print(top_k_mask.eval(feed_dict))
# print(top_k_mask.eval(feed_dict).shape)
# print("top k mask")
# print(top_k_mask_sentences.eval(feed_dict))
# print(top_k_mask_sentences.eval(feed_dict).shape)
# # print("doc sentences")
# # print(doc_sentences.eval(feed_dict))
# # print(doc_sentences.eval(feed_dict).shape)

# print("h")
# print(top_k_indices_sentences.eval(feed_dict))
# print(top_k_indices_sentences.eval(feed_dict).shape)
# print("ha")
# print(top_k_sentences.eval(feed_dict))
# print(top_k_sentences.eval(feed_dict).shape)

# print(batch_mask.eval(feed_dict))
# print(batch_mask.eval(feed_dict).shape)
# print(sentence_mask.eval(feed_dict))
# print(sentence_mask.eval(feed_dict).shape)


# print("top k")
# print(top_k_indices.eval(feed_dict))
# print(top_k_indices.eval(feed_dict).shape)
# print("top k values")
# print(top_k_values.eval(feed_dict))
# print(top_k_values.eval(feed_dict).shape)

# print(max_sent_per_doc.eval(feed_dict))
# print(max_sent_per_doc.eval(feed_dict).shape)
# print("tile")
# print(tiletest.eval(feed_dict))
# print(tiletest.eval(feed_dict).shape)

"""
# print(sentence_embeddings.eval(feed_dict))
# print(max_doc_length.eval(feed_dict))
# print(sentence_embeddings.eval(feed_dict).shape)

# print(masked_sentences.eval(feed_dict))
# print("cbow sentences")
# print(cbow_sentences.eval(feed_dict))
# print(cbow_sentences.eval(feed_dict).shape)
print("doc sentences")
print(doc_sentences.eval(feed_dict))
print(doc_sentences.eval(feed_dict).shape)
# print(masked_question.eval(feed_dict))
# print(masked_question.eval(feed_dict).shape)
print("question cbow")
print(question_cbow.eval(feed_dict))
print(question_cbow.eval(feed_dict).shape)
print("alpha weights")
print(dot_prod.eval(feed_dict))
print(dot_prod.eval(feed_dict).shape)
print("topk")

"""

sess.close()
