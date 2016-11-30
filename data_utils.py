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

def one_time_data_preparation():

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

    print("Number of training questions: ", len(train_q))

    # Validation
    with open('../data_prep/top_50k/val_questions.txt', 'r') as val_q_file:
        val_q = [x.strip() for x in val_q_file.readlines()]

    print ("Number of validation questions: ", len(val_q))


    # Test
    with open('../data_prep/top_50k/test_questions.txt', 'r') as test_q_file:
        test_q = [x.strip() for x in test_q_file.readlines()]

    print ("Number of test questions: ", len(test_q))


    # Build documents vocabulary
    all_corpus_d = train_d + val_d + test_d

    # Build questions vocabulary
    all_corpus_q = train_q + val_q + test_q

    # Preparing the answers

    # Train
    with open('../data_prep/top_50k/train_choices_ent.txt', 'r') as train_choice_file:
        all_train_choices = [x.strip().replace(",", ' ') for x in train_choice_file.readlines()]

    with open('../data_prep/top_50k/train_correct_choices_ent.txt', 'r') as train_correct_file:
        train_correct_choices = [x.strip() for x in train_correct_file.readlines()]

    # Validation
    with open('../data_prep/top_50k/val_choices_ent.txt', 'r') as val_choice_file:
        all_val_choices = [x.strip().replace(",", ' ') for x in val_choice_file.readlines()]

    with open('../data_prep/top_50k/val_correct_choices_ent.txt', 'r') as val_correct_file:
        val_correct_choices = [x.strip() for x in val_correct_file.readlines()]

    # Test
    with open('../data_prep/top_50k/test_choices_ent.txt', 'r') as test_choice_file:
        all_test_choices = [x.strip().replace(",", ' ') for x in test_choice_file.readlines()]

    with open('../data_prep/top_50k/test_correct_choices_ent.txt', 'r') as test_correct_file:
        test_correct_choices = [x.strip() for x in test_correct_file.readlines()]

    all_choices = all_test_choices + all_val_choices + all_train_choices
    max_choices = max([len(x.split(" ")) for x in all_choices])
    #vocab_processor_choices = learn.preprocessing.VocabularyProcessor(max_choices)

    all_corpus = all_corpus_d + all_corpus_q

    max_vocab = max([len(x.split(" ")) for x in all_corpus])

    # Used only for the first time and the data is stored in pickle files for using later.

    all_corpus_vocabulary = learn.preprocessing.VocabularyProcessor(max_vocab)
    # Saving the vocabulary for future purposes
    cPickle.dump(all_corpus_vocabulary, open('pickled_data/all_corpus_vocab.p', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    x_train_d = np.array(list(all_corpus_vocabulary.fit_transform(train_d)))
    np.save(open('pickled_data/x_train_d', 'wb'), x_train_d)

    x_val_d = np.array(list(all_corpus_vocabulary.fit_transform(val_d)))
    np.save(open('pickled_data/x_val_d', 'wb'), x_val_d)

    x_test_d = np.array(list(all_corpus_vocabulary.fit_transform(test_d)))
    np.save(open('pickled_data/x_test_d', 'wb'), x_test_d)

    x_train_q = np.array(list(all_corpus_vocabulary.fit_transform(train_q)))
    np.save(open('pickled_data/x_train_q', 'wb'), x_train_q)

    x_val_q = np.array(list(all_corpus_vocabulary.fit_transform(val_q)))
    np.save(open('pickled_data/x_val_q', 'wb'), x_val_q)

    x_test_q = np.array(list(all_corpus_vocabulary.fit_transform(test_q)))
    np.save(open('pickled_data/x_test_q', 'wb'), x_test_q)

    y_train_choices = np.array(list(all_corpus_vocabulary.fit_transform(all_train_choices)))
    np.save(open('pickled_data/y_train_choices', 'wb'), y_train_choices)

    y_val_choices = np.array(list(all_corpus_vocabulary.fit_transform(all_val_choices)))
    np.save(open('pickled_data/y_val_choices', 'wb'), y_val_choices)

    y_test_choices = np.array(list(all_corpus_vocabulary.fit_transform(all_test_choices)))
    np.save(open('pickled_data/y_test_choices', 'wb'), y_test_choices)

    y_train = np.array(list(all_corpus_vocabulary.fit_transform(train_correct_choices)))
    np.save(open('pickled_data/y_train', 'wb'), y_train)

    y_val = np.array(list(all_corpus_vocabulary.fit_transform(val_correct_choices)))
    np.save(open('pickled_data/y_val', 'wb'), y_val)

    y_test = np.array(list(all_corpus_vocabulary.fit_transform(test_correct_choices)))
    np.save(open('pickled_data/y_test', 'wb'), y_test)


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
    if pickle_path:
        #pickle.dump(vocab_dict, open(os.path.join(pickle_path, "vocabulary_dict"), "wb"))
        pickle.dump(vocab_dict, open("vocabulary_dict.p", "wb"))
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
    entity_counts = []
    # Marks whether entity in the document
    #entity_mask = np.zeros((len(answers), len(entity_dict)))

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
            entity_count = 0
            for w in d_words:
                if w in entity_dict:
                    num = int(w[7:])
                    if num > entity_count:
                        entity_count = num
            entity_counts.append(entity_count)
            #print(d_words)
            #print(entity_count)
            #print([entity_dict[w] for w in d_words if w in entity_dict])
            #entity_mask[i, [entity_dict[w] for w in d_words if w in entity_dict]] = 1
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
    return (d_indices, q_indices, a_indices, entity_counts)

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


def make_batches(num_epochs, batch_size, shuffle, dataset, data_path, max_words, max_examples=None):
    file_path = os.path.join(data_path, dataset+".txt")
    documents, questions, answers = load_data(file_path, max_examples=max_examples)
    if dataset == "train":
        vocabulary_dict = build_vocab(documents+questions, max_words=max_words)
        entity_markers = list(set([w for w in vocabulary_dict.keys() if w.startswith('@entity')] + answers))
        entity_markers = sorted(entity_markers)
        entity_markers = ['<unk_entity>'] + entity_markers
        entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
        pickle.dump(entity_dict, open("entity.p", "wb"))
    else:
        vocabulary_dict = pickle.load(open("vocabulary_dict.p", "rb"))
        entity_dict = pickle.load(open("entity.p", "rb"))
    # assuming num entities in dev and test <= num entities in train

    d_indices, q_indices, a_indices, entity_counts = \
        vectorize_data(documents, questions, answers, vocabulary_dict, entity_dict)
    train_data = list(zip(d_indices, q_indices, a_indices, entity_counts))
    batches = batch_iter(train_data, num_epochs=num_epochs, batch_size=batch_size, shuffle=shuffle)
    return batches

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
