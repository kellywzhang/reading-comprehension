import numpy as np
import re

cutoff = 200

def split_sentences(document):
    sent_delimit = {".":1, ";":2, "?":3, "!":4}

    all_words = []
    word = ""
    all_sentences = []
    sentence = []
    for char in document:
        if char == " ":
            all_words.append(word)
            if word != "":
                sentence.append(word)
            word = ""
        elif char in sent_delimit.keys():
            all_words.append(word)
            all_words.append(char)

            if word != "":
                sentence.append(word)
            sentence.append(char)
            all_sentences.append(sentence)
            word = ""
            sentence = []
        else:
            word += char

    if len(all_sentences) > 0:
        all_sentences[-1].append(word[:-1])
    elif len(word) > 0:
        all_sentences.append(word[:-1])

    return all_sentences

def split_words(question):
    words = question.split(" ")
    words[-1] = words[-1][:-1]
    return words

def write_sentences(all_sentences, name):
    doc_str = ""
    for i in range(len(all_sentences)):
        doc_str += " ".join(all_sentences[i])
        doc_str += " "

    doc_str = doc_str[:-1] + "\n"
    f = open(name+"_documents.txt", "a")
    f.write(doc_str)
    f.close

if __name__ == '__main__':
    datasets_prefix = ["val", "test", "train"]
    index = 2

    d_file = open(datasets_prefix[index]+"_documents_old.txt", "r")

    documents = d_file.readlines()

    d_file.close()

    for i in range(len(documents)):
        dline = documents[i]

        all_sentences = split_sentences(dline)

        write_sentences(all_sentences, "updated_"+datasets_prefix[index])

        if i % 1000 == 0:
            print(i)
