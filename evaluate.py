# External library imports
import tensorflow as tf
import numpy as np
import datetime
import os
import glob

from StanfordReader import StanfordReader
from tensorflow.contrib import learn
import data_utils
import pickle

checkpoint_dir = "/Users/kellyzhang/Documents/ReadingComprehension/reading-comprehension/deploy/runs/1481986574/checkpoints"
#"/Users/kellyzhang/Documents/ReadingComprehension/reading-comprehension/deploy/runs_aws/1482035340/checkpoints"
data_path = "/Users/kellyzhang/Documents/ReadingComprehension/reading-comprehension/deploy/data/"

# Load Test Data
# =================================================================================
test_batch = data_utils.create_batches_wdw(num_epochs=1, batch_size=10000,\
    shuffle=False, data_path=data_path, dataset="test", num_examples=10000, vocab_size=50000)

for batch in test_batch:
    d_padded, q_padded, c_indices, a_indices = data_utils.pad_batch_wdw(batch, train=False)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()

    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_d = graph.get_operation_by_name("input_d").outputs[0]
        input_q = graph.get_operation_by_name("input_q").outputs[0]
        input_a = graph.get_operation_by_name("input_a").outputs[0]
        input_m = graph.get_operation_by_name("input_m").outputs[0]

        # Tensors we want to evaluate
        correct_vector = graph.get_operation_by_name("prediction/correct_vector").outputs[0]

        # Collect the predictions here
        all_predictions = []
        all_labels = []

        correct = 0
        # Generate batches for one epoch
        result = sess.run([correct_vector], {input_d: d_padded, input_q: q_padded, input_a: c_indices, input_m: a_indices})
        print(result)
        print(np.mean(result[0]))
