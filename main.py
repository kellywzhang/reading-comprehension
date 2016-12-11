"""
Goal:
    - Create batches of (document, question, answer) tuples to feed into NN
    - Create a vocabulary dictionary that can be referred to later
    - Run StanfordReader with batches
    - Save model loss, variables, etc.

Datasets:
    CNN (http://cs.nyu.edu/~kcho/DMQA/)
        Train:      380,298
        Validation: 3,924
        Test:       3,198
    DailyMail (http://cs.nyu.edu/~kcho/DMQA/)
    Who-Did-What

TODO/ISSUES: Numbers/times in documents (not represented well in vocabulary)

Credits: Primarily adapted from https://github.com/danqi/rc-cnn-dailymail
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import datetime
import pickle

import data_utils
import tf_helpers
from StanfordReader import StanfordReader

# Code based on: https://github.com/dennybritz/cnn-text-classification-tf

vocab_size=10000
batch_size=32
num_epochs=5
max_global_norm=10
max_entities = len(pickle.load(open("entity.p", "rb")))
print(max_entities)

# Load Data
# =================================================================================
data_path = "/Users/kellyzhang/Documents/ReadingComprehension/DeepMindDataset/cnn/questions"
batches = data_utils.make_batches(num_epochs=num_epochs, batch_size=batch_size, shuffle=False, dataset="train", data_path=data_path, max_words=vocab_size, max_examples=500)

dev_batch = data_utils.load_data(dataset="validation", data_path=data_path, max_words=None, max_examples=32)

d_indices_dev, q_indices_dev, a_indices_dev, entity_counts_dev = \
    data_utils.pad_batch(np.array(dev_batch), train=False)

# Helper Functions
# =================================================================================
def train_step(data, current_step, writer=None, print_bool=False):
    """
    Single training step
    """
    d_indices, q_indices, a_indices, entity_counts = data
    feed_dict = {
        stan_reader.input_d: d_indices,
        stan_reader.input_q: q_indices,
        stan_reader.input_a: a_indices,
        stan_reader.input_m: entity_counts
    }
    # print("hidden states d")
    # attending = stan_reader.attention.attending_tensor.eval(feed_dict)
    # attended = stan_reader.attention.attended.eval(feed_dict)
    # dot_prod = stan_reader.attention.dot_prod.eval(feed_dict)
    # print("attending")
    # print(attending)
    # print("attended")
    # print(attended)
    # print("dot prod")
    # print(dot_prod)
    # print("attending")
    # print(attending.shape)
    # print("attended")
    # print(attended.shape)
    # print("dot prod")
    # print(dot_prod.shape)
    # print(attending[0].shape)
    # print(attended[0][0].shape)
    # print(np.dot(np.squeeze(attending[0]), attended[0][0]))
    # print("alpha weights")
    # print(stan_reader.alpha_weights.eval(feed_dict))
    # print("attended weighted")
    # print(stan_reader.attention.attended_weighted.eval(feed_dict).shape)

    #print(np.sum)
    # print("answer")
    # print(stan_reader.input_a.eval(feed_dict))
    # print("predict-probs")
    # print(stan_reader.predict_probs.eval(feed_dict))
    # print("predict_probs noramlized")
    # print(stan_reader.predict_probs_normalized.eval(feed_dict))
    # print("likelihoods")
    # print(stan_reader.likelihoods.eval(feed_dict))
    _, summaries, loss_val, accuracy_val = sess.run([train_op, train_summary_op, stan_reader.loss, stan_reader.accuracy], feed_dict)

    time_str = datetime.datetime.now().isoformat()
    if print_bool:
        print("\nTrain: {}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, loss_val, accuracy_val))
    if writer:
        writer.add_summary(summaries, current_step)

    return (loss_val, accuracy_val)

def dev_step(data, current_step, writer=None):
    """
    Evaluates model on a validation set
    """
    d_indices_dev, q_indices_dev, a_indices_dev, entity_counts_dev = data

    feed_dict = {
        stan_reader.input_d: d_indices_dev,
        stan_reader.input_q: q_indices_dev,
        stan_reader.input_a: a_indices_dev,
        stan_reader.input_m: entity_counts_dev
    }

    summaries, loss_val, accuracy_val = sess.run([dev_summary_op, stan_reader.loss, stan_reader.accuracy], feed_dict)

    time_str = datetime.datetime.now().isoformat()
    print("Dev:   {}: step {}, loss {:g}, acc {:g}".format(time_str, current_step, loss_val, accuracy_val))
    if writer:
        writer.add_summary(summaries, current_step)

    return (loss_val, accuracy_val)

# Starting Session
# ================================================================================
sess = tf.InteractiveSession()
stan_reader = StanfordReader(
        max_entities=max_entities,
        hidden_size=128,
        vocab_size=vocab_size,
        embedding_dim=100,
        batch_size=batch_size
    )

#optimizer = tf.train.AdamOptimizer(0.0001)
optimizer = tf.train.GradientDescentOptimizer(0.1)
global_step = tf.Variable(0, name='global_step', trainable=False)
trainables = tf.trainable_variables()
grads = tf.gradients(stan_reader.loss, trainables)
grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
grad_var_pairs = zip(grads, trainables)

# capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(grad_var_pairs, global_step=global_step)

train_summary_op, dev_summary_op, train_summary_writer, dev_summary_writer, timestamp, checkpoint_prefix = \
    tf_helpers.save_summaries(sess, [stan_reader.loss, stan_reader.accuracy], grad_var_pairs)
saver = tf.train.Saver(tf.all_variables())

# Training and Validation
# ===============================================================================
sess.run(tf.initialize_all_variables())

def loss_early_stopping():
    min_loss = 999999999
    increasing_loss_count = 0
    max_accuracy = 0
    max_accuracy_step = 0

    for batch in batches:
        x_batch, y_batch = zip(*batch) # TODO: SETUP YOUR DATA'S BATCHES

        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            train_loss, train_accuracy = train_step(x_batch, y_batch, current_step, print_bool=True)
            dev_loss, dev_accuracy = dev_step(x_dev, y_dev, current_step)

            if dev_loss < min_loss:
                min_loss = dev_loss
                increasing_loss_count = 0
            else:
                increasing_loss_count += 1

            if dev_accuracy > max_accuracy:
                max_accuracy = dev_accuracy
                max_accuracy_step = current_step

            if current_step > FLAGS.patience and FLAGS.patience_increase < increasing_loss_count:
                break

        else:
            train_loss, train_accuracy = train_step(x_batch, y_batch, current_step, print_bool=False)

        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=global_step)
            print("Saved model checkpoint to {}".format(path))

    return (train_loss, train_accuracy, max_accuracy, max_accuracy_step)

def accuracy_early_stopping():
    max_accuracy = 0
    max_accuracy_step = 0

    #for batch in batches:
        #x_batch, y_batch = zip(*batch) # TODO: SETUP YOUR DATA'S BATCHES

    for _ in range(1000):
        x_batch, y_batch = mnist.train.next_batch(FLAGS.batch_size)

        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            train_loss, train_accuracy = train_step(x_batch, y_batch, current_step, print_bool=True)
            dev_loss, dev_accuracy = dev_step(x_dev, y_dev, current_step)

            if dev_accuracy > max_accuracy:
                max_accuracy = dev_accuracy
                max_accuracy_step = current_step

            if current_step > FLAGS.patience and FLAGS.patience_increase < current_step - max_accuracy_step:
                break

        else:
            train_loss, train_accuracy = train_step(x_batch, y_batch, current_step, print_bool=False)

        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=global_step)
            print("Saved model checkpoint to {}".format(path))

    return (train_loss, train_accuracy, max_accuracy, max_accuracy_step)

def run_for_epochs(batches):
    for batch in batches:
        d_indices, q_indices, a_indices, entity_counts = data_utils.pad_batch(batch)

        current_step = tf.train.global_step(sess, global_step)
        if current_step % 5 == 0:
            train_loss, train_accuracy = \
                train_step((d_indices, q_indices, a_indices, entity_counts), current_step, print_bool=True)
            dev_loss, dev_accuracy = \
                dev_step((d_indices_dev, q_indices_dev, a_indices_dev, entity_counts_dev), current_step, writer=dev_summary_writer)


        else:
            train_loss, train_accuracy = \
                train_step((d_indices, q_indices, a_indices, entity_counts), current_step, print_bool=False)

        if current_step % 5 == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=global_step)
            print("Saved model checkpoint to {}".format(path))

    return (train_loss, train_accuracy)

train_loss, train_accuracy = run_for_epochs(batches)

print("\nFinal Valildation Evaluation:")
current_step = tf.train.global_step(sess, global_step)
dev_loss, dev_accuracy = \
    dev_step((d_indices_dev, q_indices_dev, a_indices_dev, entity_counts_dev), current_step, writer=dev_summary_writer)
#print("Maximum validation accuracy at step {}: {}".format(max_accuracy_step, max_accuracy))
print("")

tf_helpers.write_results(current_step, train_loss, train_accuracy, dev_loss, dev_accuracy, timestamp)

sess.close()
