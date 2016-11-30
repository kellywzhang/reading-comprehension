'''
This module is for extracting the documents, questions, choices and correct answer for all the elements present in the data.

The specific XML file to be parsed is given as a parameter. 

Input:
	- XML file name to be parsed

** Have a directory called 'data' in the current directory to store the output of the program.

Output:
	4 files pertaining to the XML file written to the directory called 'data'. 
	4 documents will be:
		- documents
		- questions
		- choices
		- correct choice 

As part of the preprocessing, each given choice is replaced with a token taking the template @entityN,
where N is a number typically from 1 - 5 as there are a maximum of 5 choices for all questions. 

Consequently, all the choices(entities) anonymized with the token @entityN are also anonymized in the corresponding documents and questions. 
Many a times, the entities will be addressed by their full names and their first/middle/last names. To make the anonymization more robust, 
the first/middle/last names of each entity are also replaced by the same entity token.

Each entity that has to be answered in a given question is replaced by '|||', a sequence of 3 pipes. 
'''


from bs4 import BeautifulSoup
import re
import sys

train_contents = open(sys.argv[1]).read()

corpus = BeautifulSoup(train_contents)

all_comprehensions = corpus.findAll('mc')

documents_file = open('data/' + sys.arg[1][: sys.argv[1].index('.')] + '_documents.txt', 'w')
questions_file = open('data/' + sys.arg[1][: sys.argv[1].index('.')] + '_questions.txt', 'w')
choices_file = open('data/' + sys.arg[1][: sys.argv[1].index('.')] + '_choices_ent.txt', 'w')
correct_answers = open('data/' + sys.arg[1][: sys.argv[1].index('.')] + '_correct_choices_ent.txt', 'w')


for comp in all_comprehensions:
    
    document = comp.contextart.get_text().strip()
    question = comp.question.leftcontext.get_text().strip() + ' ' + comp.question.leftblank.get_text().strip() + ' ||| ' + \
                comp.question.rightblank.get_text().strip() + ' ' + comp.question.rightcontext.get_text().strip()
    
    correct_choice = ""
    all_choices = []
    for choice in comp.findAll('choice'):
        all_choices.append(choice.get_text().strip())
        if choice['correct'] == 'true':
            correct_choice = choice.get_text().strip()
    
    entity_num = 0
    entities_list = {}
    for c in all_choices:
        entities_list[c] = '@entity' + str(entity_num)
        
        all_possibilities = [c] + c.split(' ')
        for pos in all_possibilities:
            document = re.sub('(\s?' + pos + '\s|\s' + pos + '\s?)', ' @entity' + str(entity_num) + ' ', document)
            question = re.sub('(\s?' + pos + '\s|\s' + pos + '\s?)', ' @entity' + str(entity_num) + ' ', question)
        
        entity_num += 1
    
    # Check the correctness of this line
    documents_file.write(document.strip().encode('utf-8') + '\n') 
    questions_file.write(question.strip().encode('utf-8') + '\n')
    choices_file.write(','.join(entities_list.values()).strip().encode('utf-8') + '\n')
    correct_answers.write(entities_list[correct_choice].strip().encode('utf-8') + '\n')
    
documents_file.close()
questions_file.close()
choices_file.close()
correct_answers.close()