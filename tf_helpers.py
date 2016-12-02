import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import time
import datetime
import os

def save_summaries(sess, summary_var_list, grads_and_vars=None, FLAGS=None, timestamp=None):
	"""
	Creates train and validation summary writers for TensorBoard; Also creates checkpoints
	"""
	# Keep track of gradient values and sparsity
	if grads_and_vars:
		grad_summaries = []
		for g, v in grads_and_vars:
			if g is not None:
				grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
				sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		if len(grad_summaries) > 0:
			grad_summaries_merged = tf.merge_summary(grad_summaries)

	# Output directory for models and summaries
	if not timestamp:
		timestamp = str(int(time.time()))
	out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
	print("Writing to {}\n".format(out_dir))

	# Summaries for loss and accuracy
	loss_summary = tf.scalar_summary("loss", summary_var_list[0])
	acc_summary = tf.scalar_summary("accuracy", summary_var_list[1])

	# Train Summaries
	train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged]) if grads_and_vars and len(grad_summaries) > 0 else tf.merge_summary([loss_summary, acc_summary])
	train_summary_dir = os.path.join(out_dir, "summaries", "train")
	train_summary_writer = tf.train.SummaryWriter(logdir=train_summary_dir, graph=sess.graph)

	# Dev summaries
	dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
	dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
	dev_summary_writer = tf.train.SummaryWriter(logdir=dev_summary_dir, graph=sess.graph)

	# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
	checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
	checkpoint_prefix = os.path.join(checkpoint_dir, "model")
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	# Save parameters
	if FLAGS:
		with open(os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp, "Parameters.txt")), "w") as text_file:
			for attr, value in sorted(FLAGS.__flags.items()):
				text_file.write("{}={}\n".format(attr.upper(), value))

	return (train_summary_op, dev_summary_op, train_summary_writer, dev_summary_writer, timestamp, checkpoint_prefix)

def write_results(final_step, train_loss, train_accuracy, dev_loss, dev_accuracy, timestamp):
	"""
	Appends final model results to 'Parameters.txt' file
	"""
	with open(os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp, "Paramters.txt")), "a") as text_file:
		text_file.write("\n\nTime Step = {}".format(final_step))
		text_file.write("\nTrain Loss = {}".format(train_loss))
		text_file.write("\nTrain Accuracy = {}".format(train_accuracy))
		text_file.write("\nValidation Loss = {}".format(dev_loss))
		text_file.write("\nValidation Accuracy = {}".format(dev_accuracy))
