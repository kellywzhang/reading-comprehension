'''
This module is to select only the top 50k tokens (words separated by space on either side except for the words at the beginning and end of sentence).

The tokens that are not in the top 50k are replaced by a string 'oov'.
'''

import os
import re
import sys

def clean_input(string):
	string = re.sub(r"[^@|A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)

	return string


word_count = {}

all_train_corpus = 'all_train_corpus.txt'
all_files = os.listdir('data')

print "Starting to count in train files"

with open(all_train_corpus, 'r') as f:
	all_text = f.readlines()
	for actual_text in all_text:

		preprocessed = clean_input(actual_text.strip())

		for w in preprocessed.split():
			try:
				word_count[w] += 1
			except KeyError as ke:
				word_count[w] = 1


print "Lenght of word count dictionary: ", len(word_count)

k = 1
top_50k = []

for w in sorted(word_count, key = word_count.get, reverse = True):
	top_50k.append(w)

	k += 1

	if k == 50000:
		break

for tf in all_files:
	if 'documents' in tf or 'question' in tf:
		pf = open('data_prep/top_50k/' + tf, 'w+')

		with open('data/' + tf, 'r') as f:
			all_text = f.readlines()
			for actual_text in all_text:
				preprocessed = clean_input(actual_text.strip())

				#filtered_sentence = ' '.join([w if (w in top_50k) or (w == '|||') else 'oov' for w in preprocessed.split()])	
				filtered_sentence = ''
				for w in preprocessed.split():
					if w in top_50k:
						filtered_sentence += (w + ' ')
					else:
						filtered_sentence += 'oov '	
	
				pf.write(filtered_sentence.strip() + '\n')
		pf.close()
