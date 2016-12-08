#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from SentenceClassifier import SentenceClassifier
from tensorflow.contrib import learn
import data_utils
import cPickle

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# ======================== MODEL HYPERPARAMETERS ========================================
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_nodes", 16, "Number of nodes in fully connected layer")
tf.flags.DEFINE_float("learning_rate", 10**-6, "Learning rate")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "Weight lambda on l2 regularization")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
#tf.flags.DEFINE_integer("patience", 800, "Minimum number of batches seen before early stopping")
#tf.flags.DEFINE_integer("patience_increase", 6, "Number of dev evaluations of increasing loss before early stopping")

# Display/Saving Parameters
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Print
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# =============================== PREPARING DATA FOR TRAINING/VALIDATION/TESTING ===============================================
print("Loading data...")


# Loading all data points from pickle files
all_corpus_vocabulary = cPickle.load(open('/scratch/vdn207/qa_project/final_saved_data/all_corpus_vocab.p', 'rb'))

print ("Loading documents....")

x_train_d = np.load(open('/scratch/vdn207/qa_project/final_saved_data/x_train_d', 'rb'))
x_val_d = np.load(open('/scratch/vdn207/qa_project/final_saved_data/x_val_d', 'rb'))
#x_test_d = np.load(open('/scratch/vdn207/qa_project/final_saved_data/x_test_d', 'rb'))

print ("Loading questions....")

x_train_q = np.load(open('/scratch/vdn207/qa_project/final_saved_data/x_train_q', 'rb'))
x_val_q = np.load(open('/scratch/vdn207/qa_project/final_saved_data/x_val_q', 'rb'))
#x_test_q = np.load(open('/scratch/vdn207/qa_project/final_saved_data/x_test_q', 'rb'))

print ("Loading choices....")
y_train_choices = np.load(open('/scratch/vdn207/qa_project/final_saved_data/y_train_choices', 'rb'))
y_val_choices = np.load(open('/scratch/vdn207/qa_project/final_saved_data/y_val_choices', 'rb'))
#y_test_choices = np.load(open('/scratch/vdn207/qa_project/final_saved_data/y_test_choices', 'rb'))

print ("Loading correct choices....")
y_train = np.load(open('/scratch/vdn207/qa_project/final_saved_data/y_train', 'rb'))
y_val = np.load(open('/scratch/vdn207/qa_project/final_saved_data/y_val', 'rb'))
#y_test = np.load(open('/scratch/vdn207/qa_project/final_saved_data/y_test', 'rb'))

print ("Train D: ", x_train_d.shape)
print ("Val D: ", x_val_d.shape)
#print ("Test D: ", x_test_d.shape)
print ("Train Q: ", x_train_q.shape)
print ("Val Q: ", x_val_q.shape)
#print ("Test Q: ", x_test_q.shape)


batch_accuracy_training = []
val_set_accuracy = []

# ================================================== MODEL TRAINING ======================================

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
	allow_soft_placement=FLAGS.allow_soft_placement,
	log_device_placement=FLAGS.log_device_placement)

	session_conf.gpu_options.allow_growth = True

	sess = tf.Session(config=session_conf)

	with sess.as_default():

        # FIX THISS
		sent_classifier = SentenceClassifer(max_entities = 5, batch_size = FLAGS.batch_size)

		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
		grads_and_vars = optimizer.compute_gradients(sent_classifier.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


		# Keep track of gradient values and sparsity (optional)
		grad_summaries = []
		for g, v in grads_and_vars:
			if g is not None:
			    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
			    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
			    grad_summaries.append(grad_hist_summary)
			    grad_summaries.append(sparsity_summary)
			grad_summaries_merged = tf.merge_summary(grad_summaries)



		# Output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		print("Writing to {}\n".format(out_dir))

		# Summaries for loss and accuracy
		loss_summary = tf.scalar_summary("loss", sent_classifier.loss)
		acc_summary = tf.scalar_summary("accuracy", sent_classifier.accuracy)

		# Train Summaries
		train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

		# Dev summaries
		dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

		# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.all_variables())

		# Write vocabulary
		all_corpus_vocabulary.save(os.path.join(out_dir, "vocab"))

		# Initialize all variables
		sess.run(tf.initialize_all_variables())


		def train_step(x_batch_d, x_batch_q, y_batch_choices, y_batch):

			seq_len_d = np.array([np.sum(doc != 0) for doc in x_batch_d])
			seq_len_q = np.array([np.sum(ques != 0) for ques in x_batch_q])
			max_seq_len_d = np.max(seq_len_d)
			max_seq_len_q = np.max(seq_len_q)

			#A single training step
			feed_dict = {
				sent_classifier.seq_lens_d : seq_len_d,
			    sent_classifier.seq_lens_q : seq_len_q,
			    sent_classifier.input_d : tuple([doc[: max_seq_len_d] for doc in x_batch_d]),
			    sent_classifier.input_q : tuple([ques[: max_seq_len_q] for ques in x_batch_q]),
			    sent_classifier.input_a : y_batch,
			    sent_classifier.input_m : np.array([np.sum(c != 0) for c in y_batch_choices]),
			}

			#print ("Ready for training....")

			_, step, loss, accuracy = sess.run(
			    [train_op, global_step, sent_classifier.loss, sent_classifier.accuracy],
			    feed_dict)

			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

			train_summary_writer.add_summary(summaries, step)

		def dev_step(x_val_d, x_val_q, y_val_choices, y_val, writer=None):

			seq_len_d = np.array([np.sum(doc != 0) for doc in x_val_d])
                        seq_len_q = np.array([np.sum(ques != 0) for ques in x_val_q])
                        max_seq_len_d = np.max(seq_len_d)
                        max_seq_len_q = np.max(seq_len_q)

			# Evaluates model on a dev set
			feed_dict = {
				sent_classifier.seq_lens_d : np.array([np.sum(doc != 0) for doc in x_val_d]),
			    sent_classifier.seq_lens_q : np.array([np.sum(ques != 0) for ques in x_val_q]),
			    sent_classifier.input_d : tuple([doc[: max_seq_len_d] for doc in x_val_d]),
			    sent_classifier.input_q : tuple([ques[: max_seq_len_q] for ques in x_val_q]),
			    sent_classifier.input_a : y_val,
			    sent_classifier.input_m : np.array([np.sum(c != 0) for c in y_val_choices]),
			}
			step, summaries, loss, accuracy = sess.run(
			    [global_step, dev_summary_op, sent_classifier.loss, sent_classifier.accuracy],
			    feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			if writer:
			    writer.add_summary(summaries, step)


		# Generate batches
		batches = batch_iter(list(zip(x_train_d, x_train_q, y_train_choices, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

		for batch in batches:

			x_batch_d, x_batch_q, y_batch_choices, y_batch = zip(*batch)

			batch_accuracy = train_step(x_batch_d, x_batch_q, y_batch_choices, y_batch)


			current_step = tf.train.global_step(sess, global_step)
			if current_step % FLAGS.evaluate_every == 0:
			    print("\nEvaluation:")
			    dev_step(x_val_d, x_val_q, y_val_choices, y_val, writer=dev_summary_writer)

			if current_step % FLAGS.checkpoint_every == 0:
			    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
			    print("Saved model checkpoint to {}\n".format(path))
