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

#documents, questions, answers = data_utils.load_data(train_path, max_examples=100)
batches = data_utils.make_batches(num_epochs=1, batch_size=2, shuffle=False, dataset="train", data_path=data_path, max_words=10000, max_examples=100)

d_indices_dev, q_indices_dev, a_indices_dev, entity_counts_dev = data_utils.load_data(dataset="validation", data_path=data_path, max_words=None, max_examples=100)


count = 0
for batch in batches:
    d_indices, q_indices, a_indices, entity_counts = data_utils.pad_batch(batch)

    print(d_indices)
    print(q_indices)
    print(a_indices)
    print(entity_counts)

    count+=1
    if count > 3:
        break

# Starting interactive Session
sess = tf.InteractiveSession()

# MODEL
stan_reader = StanfordReader(
		max_entities=len(pickle.load(open("entity.p", "rb"))),
        hidden_size=128,
        vocab_size=vocab_size,
        embedding_dim=100,
        batch_size=batch_size
	)



sess.run(tf.initialize_all_variables())

sess.close()
