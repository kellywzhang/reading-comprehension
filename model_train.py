#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from StanfordReader import StanfordReader
from tensorflow.contrib import learn


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
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "Weight lambda on l2 regularization")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("patience", 800, "Minimum number of batches seen before early stopping")
tf.flags.DEFINE_integer("patience_increase", 6, "Number of dev evaluations of increasing loss before early stopping")

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

# LOADING DOCUMENTS

# Train
with open('../data_prep/top_50k/train_documents.txt', 'r') as train_d_file:
	train_d = [x.strip() for x in train_d_file.readlines()]

print ("Number of training documents: ", len(train_d))

# Validation
with open('../data_prep/top_50k/val_documents.txt', 'r') as val_d_file:
	val_d = [x.strip() for x in val_d_file.readlines()]

print ("Number of validation documents: ", len(val_d))

# Test
with open('../data_prep/top_50k/test_documents.txt', 'r') as test_d_file:
	test_d = [x.strip() for x in test_d_file.readlines()]

print ("Number of test documents: ", len(test_d))


# LOADING QUESTIONS

# Train
with open('../data_prep/top_50k/train_questions.txt', 'r') as train_q_file:
	train_q = [x.strip() for x in train_q_file.readlines()]

print ("Number of training questions: ", len(train_q))

# Validation
with open('../data_prep/top_50k/val_questions.txt', 'r') as val_q_file:
	val_q = [x.strip() for x in val_q_file.readlines()]

print ("Number of validation questions: ", len(val_q))


# Test
with open('../data_prep/top_50k/test_questions.txt', 'r') as test_q_file:
	test_q = [x.strip() for x in test_q_file.readlines()]

print ("Number of test questions: ", len(test_q))


# Build documents vocabulary
train_corpus_d = train_d + val_d + test_d
max_length_d = max([len(x.split(" ")) for x in train_corpus_d])
vocab_processor_d = learn.preprocessing.VocabularyProcessor(max_length_d)

x_train_d = np.array(list(vocab_processor_d.fit_transform(train_d)))
x_val_d = np.array(list(vocab_processor_d.fit_transform(val_d)))
x_test_d = np.array(list(vocab_processor_d.fit_transform(test_d)))

print ("Size of document training set: ", x_train_d.shape)
print ("Size of document validation set: ", x_val_d.shape)
print ("Size of document test set: ", x_test_d.shape)

# Build questions vocabulary
train_corpus_q = train_q + val_q + test_q
max_length_q = max([len(x.split(" ")) for x in train_corpus_q])
vocab_processor_q = learn.preprocessing.VocabularyProcessor(max_length_q)

x_train_q = np.array(list(vocab_processor_q.fit_transform(train_q)))
x_val_q = np.array(list(vocab_processor_q.fit_transform(val_q)))
x_test_q = np.array(list(vocab_processor_q.fit_transform(test_q)))

print ("Size of question training set: ", x_train_q.shape)
print ("Size of question validation set: ", x_val_q.shape)
print ("Size of question test set: ", x_test_q.shape)


# Preparing the answers

# Train
with open('../data_prep/top_50k/train_choices.txt', 'r') as train_choice_file:
	all_train_choices = [y for y in x.strip().split(',') for x in train_choice_file.readlines()]

with open('../data_prep/top_50k/train_correct_choices.txt', 'r') as train_correct_file:
	train_correct_choices = [x.strip() for x in train_correct_file.readlines()]

# Validation
with open('../data_prep/top_50k/val_choices.txt', 'r') as val_choice_file:
	all_val_choices = [y for y in x.strip().split(',') for x in val_choice_file.readlines()]

with open('../data_prep/top_50k/val_correct_choices.txt', 'r') as val_correct_file:
	val_correct_choices = [x.strip() for x in val_correct_file.readlines()]

# Test
with open('../data_prep/top_50k/test_choices.txt', 'r') as test_choice_file:
	all_test_choices = [y for y in x.strip().split(',') for x in test_choice_file.readlines()]

with open('../data_prep/top_50k/test_correct_choices.txt', 'r') as test_correct_file:
	test_correct_choices = [x.strip() for x in test_correct_file.readlines()]

all_choices = set(all_test_choices + all_val_choices + all_train_choices)
all_choices_corpus = list(all_choices)
vocab_processor_choices = learn.preprocessing.VocabularyProcessor(len(all_choices_corpus))

y_train_choices = np.array(list(vocab_processor_choices.fit_transform(all_train_choices)))
y_val_choices = np.array(list(vocab_processor_choices.fit_transform(all_val_choices)))
y_test_choices = np.array(list(vocab_processor_choices.fit_transform(all_test_choices)))

y_train = np.array(list(vocab_processor_choices.fit_transform(train_correct_choices)))
y_val = np.array(list(vocab_processor_choices.fit_transform(val_correct_choices)))
y_test = np.array(list(vocab_processor_choices.fit_transform(test_correct_choices)))


batch_accuracy_training = []
val_set_accuracy = []

# ================================================== MODEL TRAINING ======================================

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
	allow_soft_placement=FLAGS.allow_soft_placement,
	log_device_placement=FLAGS.log_device_placement)

	sess = tf.Session(config=session_conf)

	with sess.as_default():
		
		stan_reader = StanfordReader(max_entities = len(all_choices_corpus))
		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.0005)
		grads_and_vars = optimizer.compute_gradients(stan_reader.loss)
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
		loss_summary = tf.scalar_summary("loss", stan_reader.loss)
		acc_summary = tf.scalar_summary("accuracy", stan_reader.accuracy)

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
		vocab_processor.save(os.path.join(out_dir, "vocab"))

		# Initialize all variables
		sess.run(tf.initialize_all_variables())

		
		def train_step(x_batch_d, x_batch_q, y_batch_choices, y_batch):
			#A single training step
			feed_dict = {
			    stan_reader.seq_lens_d : np.array([np.sum(doc != 0) for doc in x_batch_d]),
			    stan_reader.seq_lens_q : np.array([np.sum(ques != 0) for ques in x_batch_q]),
			    stan_reader.input_d : x_batch_d,
			    stan_reader.input_q : x_batch_q,
			    stan_reader.input_a : y_batch,
			    stan_reader.input_m : np.array([np.sum(c != 0) for c in y_batch_choices])
			}
			_, step, summaries, loss, accuracy = sess.run(
			    [train_op, global_step, train_summary_op, stan_reader.loss, stan_reader.accuracy],
			    feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

			train_summary_writer.add_summary(summaries, step)

		def dev_step(x_val_d, x_val_q, y_val_choices, y_val, writer=None):
			# Evaluates model on a dev set
			feed_dict = {
				stan_reader.seq_lens_d : np.array([np.sum(doc != 0) for doc in x_val_d]),
			    stan_reader.seq_lens_q : np.array([np.sum(ques != 0) for ques in x_val_q]),
			    stan_reader.input_d : x_val_d,
			    stan_reader.input_q : x_val_q,
			    stan_reader.input_a : y_val,
			    stan_reader.input_m : y_val_choices
			}
			step, summaries, loss, accuracy = sess.run(
			    [global_step, dev_summary_op, stan_reader.loss, stan_reader.accuracy],
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

