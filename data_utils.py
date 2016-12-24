"""
Goal:
    - Create batches of (document, question, answer) tuples to feed into NN
    - Create a vocabulary dictionary that can be referred to later

Datasets:
    CNN (http://cs.nyu.edu/~kcho/DMQA/)
        Train:      380,298
        Validation: 3,924
        Test:       3,198
    DailyMail (http://cs.nyu.edu/~kcho/DMQA/)
    Who-Did-What

TODO/ISSUES: Numbers/times in documents (not represented well in vocabulary)
             Make method for loading pre-trained word embeddings

Credits: Primarily adapted from https://github.com/danqi/rc-cnn-dailymail
"""

import numpy as np
import pickle
from collections import Counter
import os
from tensorflow.contrib import learn

def load_text_data_old(path, dataset, max_examples=None):
    files = ["_documents_old.txt", "_questions.txt", "_choices_ent.txt", "_correct_choices_ent.txt"]
    datasets = ["train", "val", "test"]

    assert dataset in datasets

    data = []

    for i in range(len(files)):
        f = open(path+dataset+files[i], 'r')
        num_examples = 0
        examples = []
        while True:
            line = f.readline()
            if not line:
                break

            document = line.strip().lower()
            words = document.split(" ")
            document = " ".join(words[:250])
            num_examples += 1

            examples.append(document)

            if (max_examples is not None) and (num_examples >= max_examples):
                break
        #print(examples)
        data.append(examples)
    print(len(data[0]))
    print("#Examples: {}".format(len(examples)))
    f.close()
    return data

def load_text_data(path, dataset, max_examples=None):
    files = ["_documents.txt", "_questions.txt", "_choices_ent.txt", "_correct_choices_ent.txt"]
    datasets = ["train", "val", "test"]

    assert dataset in datasets

    data = []

    for i in range(len(files)):
        f = open(path+dataset+files[i], 'r')
        num_examples = 0
        examples = []
        while True:
            line = f.readline()
            if not line:
                break

            document = line.strip().lower()
            num_examples += 1

            examples.append(document)

            if (max_examples is not None) and (num_examples >= max_examples):
                break
        #print(examples)
        data.append(examples)
    print(len(data[0]))
    print("#Examples: {}".format(len(examples)))
    f.close()
    return data

def build_vocab(sentences, pickle_path=True, max_words=50000):
    """
    Purpose:
    Builds a dict (word, index) for `max_words` number of words in `sentences`.
    All other words mapped to <UNK>; 1 used for delimiter |||.
    """
    word_count = Counter()
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)

    # leave 0 to UNK
    # leave 1 to delimiter |||
    vocab_dict = {w[0]: index for (index, w) in enumerate(ls)}
    if pickle_path:
        pickle.dump(vocab_dict, open("vocab_dict.p", "wb"))
    return vocab_dict

def vectorize_data_wdw(documents, questions, choices, answers, vocabulary_dict, entity_dict,
                    sort_by_len=False, verbose=True):
    """
    Purpose:
    Turns D/Q/A data from text strings to lists of embedding indices.

    Variables:
    in_l: whether the entity label occurs in the document.
    """
    d_indices = []
    q_indices = []
    c_indices = []
    a_indices = []
    entity_counts = []
    print(entity_dict)
    # Marks whether entity in the document
    #entity_mask = np.zeros((len(answers), len(entity_dict)))

    for i in range(len(answers)):
        d_words = documents[i].split(' ')
        q_words = questions[i].split(' ')
        c_words = choices[i].split(",")
        seq1 = [vocabulary_dict[w] if w in vocabulary_dict else vocabulary_dict['oov'] for w in d_words]
        seq2 = [vocabulary_dict[w] if w in vocabulary_dict else vocabulary_dict['oov'] for w in q_words]
        if (len(seq1) > 0) and (len(seq2) > 0):
            d_indices.append(seq1)
            q_indices.append(seq2)
            c_indices.append(len([entity_dict[x] for x in c_words]))
            a_indices.append(entity_dict[answers[i]])
            entity_count = 0
        if verbose and (i % 10000 == 0):
            print('Vectorization: processed {} / {}'.format(i, len(answers)))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(d_indices)
        d_indices = [d_indices[i] for i in sorted_index]
        q_indices = [q_indices[i] for i in sorted_index]
        a_indices = [a_indices[i] for i in sorted_index]
        entity_counts = [entity_counts[i] for i in sorted_index]
        #entity_counts[sorted_index]

    # Change a_indices to one-hot vector form
    return (d_indices, q_indices, c_indices, a_indices)

def batch_iter(data, num_epochs=30, batch_size=32, shuffle=True):
    """
    Purpose:
        Generates a batch iterator for a dataset.

    Credits: https://github.com/dennybritz/cnn-text-classification-tf
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

def pad_batch(batch, train=True):
    if train:
        d_indices = batch[:,0]
        q_indices = batch[:,1]
        a_indices = batch[:,2]
        entity_counts = batch[:,3]
    else:
        d_indices = batch[0]
        q_indices = batch[1]
        a_indices = batch[2]
        entity_counts = batch[3]

    d_len = max([len(x) for x in d_indices])
    q_len = max([len(x) for x in q_indices])

    d_padded = np.vstack(tuple([np.pad(x, (0, d_len-len(x)), "constant", constant_values=(-1)) for x in d_indices]))
    q_padded = np.vstack(tuple([np.pad(x, (0, q_len-len(x)), "constant", constant_values=(-1)) for x in q_indices]))

    return (d_padded, q_padded, a_indices, entity_counts)

def pad_batch_wdw(batch, train=True):
    #if train:
    d_indices = batch[:,0]
    q_indices = batch[:,1]
    c_indices = batch[:,2]
    a_indices = batch[:,3]
    # else:
    #     d_indices = batch[0]
    #     q_indices = batch[1]
    #     c_indices = batch[2]
    #     a_indices = batch[3]

    d_len = max([len(x) for x in d_indices])
    q_len = max([len(x) for x in q_indices])

    d_padded = np.vstack(tuple([np.pad(x, (0, d_len-len(x)), "constant", constant_values=(-1)) for x in d_indices]))
    q_padded = np.vstack(tuple([np.pad(x, (0, q_len-len(x)), "constant", constant_values=(-1)) for x in q_indices]))

    return (d_padded, q_padded, c_indices, a_indices)

def create_batches_wdw(num_epochs, batch_size, shuffle, data_path="", dataset="train", old=False, num_examples=None, vocab_size=100000):
    if num_examples is not None:
        if old:
            data = load_text_data_old(data_path, dataset, num_examples)
        else:
            data = load_text_data(data_path, dataset, num_examples)
    else:
        if old:
            data = load_text_data_old(data_path, dataset, num_examples)
        else:
            data = load_text_data(data_path, dataset, num_examples)

    # data[0] = documents, data[1] = questions, data[2] = choices_ent data[3] = correct_choices_ent

    if dataset == "train":
        vocab_dict = build_vocab(data[0]+data[1], vocab_size)
        entity_dict = {
            "@entity0":vocab_dict["@entity0"],
            "@entity1":vocab_dict["@entity1"],
            "@entity2":vocab_dict["@entity2"],
            "@entity3":vocab_dict["@entity3"],
            "@entity4":vocab_dict["@entity4"]
            }
        pickle.dump(vocab_dict, open("vocab_dict.p", "wb"))
        pickle.dump(entity_dict, open("entity_dict.p", "wb"))
    else:
        vocab_dict = pickle.load(open("vocab_dict.p", "rb"))
        entity_dict = pickle.load(open("entity_dict.p", "rb"))

    d_indices, q_indices, c_indices, a_indices = \
        vectorize_data_wdw(documents=data[0],
                           questions=data[1],
                           choices=data[2],
                           answers=data[3],
                           vocabulary_dict=vocab_dict,
                           entity_dict=entity_dict)

    train_data = list(zip(d_indices, q_indices, c_indices, a_indices))
    batches = batch_iter(train_data, num_epochs=num_epochs, batch_size=batch_size, shuffle=shuffle)
    return batches

if __name__ == "__main__":
    """
    Purpose:
        Run with argument path to data and will automatically create relabeled,
            single-file (one for train, validation, test) version of dataset.
    """
    # import sys
    # # argument is data_path to questions file of cnn/dailymail datasets
    # data_path = sys.argv[1]
    # train_path = os.path.join(data_path, "training")
    # validation_path = os.path.join(data_path, "validation")
    # test_path = os.path.join(data_path, "test")
    #
    # make_data_file(train_path, os.path.join(data_path, "train.txt"))
    # make_data_file(validation_path, os.path.join(da_path, "validation.txt"))
    # make_data_file(test_path, os.path.join(data_path, "test.txt"))

    # data = load_text_data("/Users/kellyzhang/Documents/ReadingComprehension/reading-comprehension/deploy/data/", "val", 10)
    # # data[0] = documents, data[1] = questions, data[2] = choices_ent data[3] = correct_choices_ent
    # vocab_dict = build_vocab(data[0]+data[1], 10000000)
    #
    # entity_dict = {
    #     "@entity0":vocab_dict["@entity0"],
    #     "@entity1":vocab_dict["@entity1"],
    #     "@entity2":vocab_dict["@entity2"],
    #     "@entity3":vocab_dict["@entity3"],
    #     "@entity4":vocab_dict["@entity4"]
    #     }
    #
    # d_indices, q_indices, c_indices, a_indices = \
    #     vectorize_data_wdw(documents=data[0],
    #                        questions=data[1],
    #                        choices=data[2],
    #                        answers=data[3],
    #                        vocabulary_dict=vocab_dict,
    #                        entity_dict=entity_dict)
    #
    # train_data = list(zip(d_indices, q_indices, c_indices, a_indices))
    # batches = batch_iter(train_data, num_epochs=200, batch_size=2, shuffle=True)
    #
    # #print(len(train_data))
    #
    # i = 0
    # for batch in batches:
    #     d_padded, q_padded, c_indices, a_indices = pad_batch_wdw(batch, train=True)
    #     print(i)
    #     i += 1

    # print(d_indices)
    # print(q_indices)
    # print(c_indices)
    # print(a_indices)

    data_path = "/Users/kellyzhang/Documents/ReadingComprehension/reading-comprehension/deploy/data/"
    doc = load_text_data_old(data_path, "val", max_examples=10)
