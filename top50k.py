import os
import re
from nltk.tokenize import TreebankWordTokenizer
#from nltk.corpus import stopwords
import sys
#from nltk.stem.porter import *

#cachedStopWords = stopwords.words("english")

def remove_stop_words_and_stem(string):
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

	# Quotes - another RE

	#after_sw = ' '.join([word for word in string.split() if word not in cachedStopWords])

	#after_sw = after_sw.strip().lower()

	#port_stemmer = PorterStemmer()

	#final_string = ' '.join([port_stemmer.stem(w) for w in after_sw.split()])

	return string


word_count = {}

all_train_corpus = 'all_train_corpus.txt'
all_files = os.listdir('data')

print "Starting to count in train files"

with open(all_train_corpus, 'r') as f:
	all_text = f.readlines()
	for actual_text in all_text:

		#preprocessed = remove_stop_words_and_stem(actual_text.strip())

		ptb_preprocessed = TreebankWordTokenizer().tokenize(actual_text.strip())
		for w in ptb_preprocessed:
			try:
				word_count[w] += 1
			except KeyError as ke:
				word_count[w] = 1


print "Lenght of word count dictionary: ", len(word_count)
print word_count['|||']

k = 1
top_50k = []

for w in sorted(word_count, key = word_count.get, reverse = True):
	top_50k.append(w)

	k += 1

	if k == 50000:
		break

for tf in all_files:
	if 'documents' in tf or 'question' in tf:
		pf = open('ptb_tokenizer/top_50k/' + tf, 'w+')

		with open('data/' + tf, 'r') as f:
			all_text = f.readlines()
			for actual_text in all_text:
				#preprocessed = remove_stop_words_and_stem(actual_text.strip())
				ptb_preprocessed = TreebankWordTokenizer().tokenize(actual_text.strip())

				#filtered_sentence = ' '.join([w if (w in top_50k) or (w == '|||') else 'oov' for w in preprocessed.split()])	
				filtered_sentence = ''
				for w in ptb_preprocessed:
					if w in top_50k:
						if w == '@':
							filtered_sentence += w
						else:
							filtered_sentence += (w + ' ')
					else:
						filtered_sentence += 'oov '	
	
				pf.write(filtered_sentence.strip() + '\n')
		pf.close()
