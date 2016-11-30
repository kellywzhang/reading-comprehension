import tensorflow as tf
import numpy as np
import os
import sys
os.path.abspath(os.path.curdir)[:-8]
sys.path.insert(0, os.path.abspath(os.path.curdir)[:-8])

import data_utils
from StanfordReader import StanfordReader

print(dir(data_utils))

data_path = "/Users/kellyzhang/Documents/ReadingComprehension/DeepMindDataset/cnn/questions"
train_path = os.path.join(data_path, "train.txt")

documents, questions, answers = data_utils.load_data(train_path, max_examples=10)
batches = data_utils.make_batches(num_epochs=1, batch_size=2, shuffle=False, dataset="train", data_path=data_path, max_words=10000, max_examples=100)

count = 0
for batch in batches:
    print(batch)
    print(count)

# Starting interactive Session
sess = tf.InteractiveSession()

# DATA
"""
documents, questions, answers = data_utils.load_data(train_path, max_examples=10)
vocabulary_dict = data_utils.build_vocab(documents+questions, max_words=10000)
entity_markers = list(set([w for w in vocabulary_dict.keys() if w.startswith('@entity')] + answers))
entity_markers = sorted(entity_markers)
entity_markers = ['<unk_entity>'] + entity_markers
entity_dict = {w: index for (index, w) in enumerate(entity_markers)}

d_indices, q_indices, a_indices, entity_counts = \
    data_utils.vectorize_data(documents, questions, answers, vocabulary_dict, entity_dict)

train_data = list(zip(d_indices, q_indices, a_indices, entity_counts))
batches = data_utils.batch_iter(train_data, num_epochs=1, batch_size=2, shuffle=False)
"""


# MODEL
stan_reader = StanfordReader(max_entities=5)


sess.run(tf.initialize_all_variables())

sess.close()
