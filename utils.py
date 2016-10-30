import utils
import os
from collections import Counter
import pickle

"""
Issues: numbers/times in documents
"""

data_path = '/Users/kellyzhang/Documents/ReadingComprehension/DeepMindDataset/cnn/questions'
train_path = os.path.join(data_path, "training")
validation_path = os.path.join(data_path, "validation")
test_path = os.path.join(data_path, "test")

# Adapted from https://github.com/danqi/rc-cnn-dailymail
def load_data(in_file_path, max_example=None, relabeling=True):
     """
    load CNN / Daily Mail data from {train | dev | test} directories
    relabeling: relabel the entities by their first occurence if it is True.
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

        if (max_example is not None) and (num_examples >= max_example):
            break

    f.close()
    print("#Examples: {}".format(len(documents)))
    return (documents, questions, answers)

def build_dict(sentences, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)

    # leave 0 to UNK
    # leave 1 to delimiter |||
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}
