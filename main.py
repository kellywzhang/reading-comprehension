"""
Goal:
    - Create batches of (document, question, answer) tuples to feed into NN
    - Create a vocabulary dictionary that can be referred to later
    - Run StanfordReader with batches
    - Save model loss, variables, etc.

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


import tensorflow as tf
import numpy as np
