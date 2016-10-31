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

Credits: Primarily adapted from https://github.com/danqi/rc-cnn-dailymail
"""

import numpy as np
import pickle
from collections import Counter
import os

def make_data_file(in_file_path, write_file, relabeling=True):
    """
    Purpose:
    Loads CNN / Daily Mail data from {training | validation | test} directories,
        relabels, and saves to single file.
    Assumes data in the format taken from http://cs.nyu.edu/~kcho/DMQA/.

    Args:
    relabeling: if True relabels the entities by occurence order.

    Return:
    (documents, questions, answers) tuple; each is a list (ordered).
    """
    documents = []
    questions = []
    answers = []
    num_examples = 0

    for document in os.listdir(in_file_path):
        f = open(os.path.join(in_file_path, document), 'r')

        content = f.read().splitlines()
        document = content[2].strip().lower()
        question = content[4].strip().lower()
        answer = content[6]

        if relabeling:
            q_words = question.split(' ')
            d_words = document.split(' ')
            assert answer in d_words

            entity_dict = {}
            entity_id = 0
            for word in d_words + q_words:
                if (word.startswith('@entity')) and (word not in entity_dict):
                    entity_dict[word] = '@entity' + str(entity_id)
                    entity_id += 1

            q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
            d_words = [entity_dict[w] if w in entity_dict else w for w in d_words]
            answer = entity_dict[answer]

            question = ' '.join(q_words)
            document = ' '.join(d_words)

        questions.append(question)
        answers.append(answer)
        documents.append(document)
        num_examples += 1

    print("#Examples: {}".format(len(documents)))
    f.close()

    f = open(write_file, 'w')
    for i in range(len(questions)):
        f.write(documents[i]+"\n")
        f.write(questions[i]+"\n")
        f.write(answers[i]+"\n")
    f.close()

    return (documents, questions, answers)

def load_data(in_file_path, max_examples=None):
    """
    Purpose:
    Loads CNN / Daily Mail data from {train | dev | test}.txt

    Args:
    max_examples: Limits number of examples loaded (useful for debugging).

    Return:
    (documents, questions, answers) tuple; each is a list (ordered).
    """
    documents = []
    questions = []
    answers = []
    num_examples = 0

    f = open(in_file_path, 'r')
    while True:
        line = f.readline()
        if not line:
            break

        document = line.strip().lower()
        question = f.readline().strip()
        answer = f.readline().strip().lower()
        num_examples += 1

        questions.append(question)
        answers.append(answer)
        documents.append(document)

        if (max_examples is not None) and (num_examples >= max_examples):
            break
    print("#Examples: {}".format(len(documents)))
    f.close()
    return (documents, questions, answers)

def build_vocab(sentences, pickle_path=None, max_words=50000):
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
    vocab_dict = {w[0]: index + 2 for (index, w) in enumerate(ls)}
    if pickle:
        pickle.dump(vocab_dict, open(os.path.join(pickle_path, "vocabulary_dict"), "wb"))
    return vocab_dict

def vectorize_data(documents, questions, answers, vocabulary_dict, entity_dict,
                    sort_by_len=True, verbose=True):
    """
    Purpose:
    Turns D/Q/A data from text strings to lists of embedding indices.

    Variables:
    in_l: whether the entity label occurs in the document.
    """
    d_indices = []
    q_indices = []
    a_indices = []
    # Marks whether entity in the dictionary
    entity_bools = np.zeros((len(answers), len(entity_dict)))

    for i in range(len(answers)):
        d_words = documents[i].split(' ')
        q_words = questions[i].split(' ')
        assert (answers[i] in d_words)
        seq1 = [vocabulary_dict[w] if w in vocabulary_dict else 0 for w in d_words]
        seq2 = [vocabulary_dict[w] if w in vocabulary_dict else 0 for w in q_words]
        if (len(seq1) > 0) and (len(seq2) > 0):
            d_indices.append(seq1)
            q_indices.append(seq2)
            a_indices.append(entity_dict[answers[i]] if answers[i] in entity_dict else 0)
            entity_bools[i, [entity_dict[w] for w in d_words if w in entity_dict]] = 1.0
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
        entity_bools = entity_bools[sorted_index]

    return d_indices, q_indices, a_indices, entity_bools

def batch_iter(data, num_epochs=30, batch_size=32, shuffle=True):
    """
    Purpose:
        Generates a batch iterator for a dataset.

    Credits: https://github.com/dennybritz/cnn-text-classification-tf
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
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

if __name__ == "__main__":
    """
    Purpose:
        Run with argument path to data and will automatically create relabeled,
            single-file (one for train, validation, test) version of dataset.
    """
    import sys
    # argument is data_path to questions file of cnn/dailymail datasets
    data_path = sys.argv[1]
    train_path = os.path.join(data_path, "training")
    validation_path = os.path.join(data_path, "validation")
    test_path = os.path.join(data_path, "test")

    make_data_file(train_path, os.path.join(data_path, "train.txt"))
    make_data_file(validation_path, os.path.join(da_path, "validation.txt"))
    make_data_file(test_path, os.path.join(data_path, "test.txt"))
