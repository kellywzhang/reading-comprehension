import tensorflow as tf
import numpy as np
import os
import time
import datetime
from StanfordReader import StanfordReader
from tensorflow.contrib import learn
import data_utils
import pickle
import sys

data_path = "/Users/kellyzhang/Documents/ReadingComprehension/reading-comprehension/deploy/data/"

# ======================== MODEL HYPERPARAMETERS ========================================
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "Weight lambda on l2 regularization")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_boolean("shuffle", True, "Shuffle between batches boolean")

# Display/Saving Parameters
tf.flags.DEFINE_integer("print_every", 10, "Print train step after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")

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
batches = data_utils.create_batches_wdw(num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,\
    shuffle=FLAGS.shuffle, data_path=data_path, dataset="train", old=True, num_examples=100, vocab_size=50000)

dev_batch = data_utils.create_batches_wdw(num_epochs=1, batch_size=10000, \
    shuffle=False, data_path=data_path, dataset="val", old=True, num_examples=1000, vocab_size=50000)

for batch in dev_batch:
    d_padded_val, q_padded_val, c_indices_val, a_indices_val = data_utils.pad_batch_wdw(batch, train=False)

# ================================================== MODEL TRAINING ======================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
        )
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)

    with sess.as_default():

        stan_reader = StanfordReader(max_entities=5, batch_size=FLAGS.batch_size)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        # aggregation_method is an experimental feature introduced for faster gradient computation
        grads_and_vars = optimizer.compute_gradients(stan_reader.loss, aggregation_method = 2)
        clipped_grads = []
        for g, v in grads_and_vars:
            if g is not None:
                clipped = tf.clip_by_norm(g, clip_norm=10.)
                clipped_grads.append((clipped, v))

        train_op = optimizer.apply_gradients(clipped_grads, global_step=global_step)

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

        # Initialize all variables
        sess.run(tf.initialize_all_variables())


        def train_step(train_d, train_q, train_choices, train_answer, print_bool=True):

            #A single training step
            feed_dict = {
                stan_reader.input_d : train_d,
                stan_reader.input_q : train_q,
                stan_reader.input_a : train_choices,
                stan_reader.input_m : train_answer
            }

            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, stan_reader.loss, stan_reader.accuracy],
               feed_dict)

            if print_bool:
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            train_summary_writer.add_summary(summaries, step)

        def dev_step(val_d, val_q, val_choices, val_answer, writer=None):

            # Evaluates model on a dev set
            feed_dict = {
                stan_reader.input_d : val_d,
                stan_reader.input_q : val_q,
                stan_reader.input_a : val_choices,
                stan_reader.input_m : val_answer
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, stan_reader.loss, stan_reader.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches

        for batch in batches:
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.print_every == 0:
                print_bool = True
            else:
                print_bool = False

            d_padded, q_padded, c_indices, a_indices = data_utils.pad_batch_wdw(batch, train=True)
            train_step(d_padded, q_padded, c_indices, a_indices, print_bool=print_bool)

            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")

                #dev_step(d_padded_val, q_padded_val, c_indices_val, a_indices_val, writer=dev_summary_writer)


            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
