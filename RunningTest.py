

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import time
import re
from sklearn.model_selection import train_test_split




# Add special tokens to vocab_to_int
#vocab_to_int = {'6': 39, '>': 53, ']': 60, 'B': 12, '2': 31, 'L': 17, '?': 44, '5': 30, '0': 32, '(': 54, '$': 48, 'S': 20, '4': 38, 'R': 5, "'": 46, 'D': 22, 'V': 26, 'E': 2, 'C': 8, 'T': 0, 'G': 9, 'K': 13, 'X': 43, '[': 59, 'U': 10, 'P': 4, '\r': 23, 'W': 24, ':': 29, '-': 28, 'Q': 21, '&': 49, '1': 36, 'M': 25, 'I': 16, '"': 47, '\t': 57, '~': 50, '^': 58, 'F': 14, '3': 40, 'H': 1, '.': 27, '/': 37, ',': 18, 'O': 6, ';': 42, ' ': 3, '!': 45, 'A': 15, '8': 34, '|': 51, 'J': 7, ')': 55, '\x0c': 56, '<': 52, '9': 35, '7': 33, 'N': 11, 'Y': 19, 'Z': 41}
vocab_to_int = {'T': 0, 'H': 1, 'E': 2, ' ': 3, 'P': 4, 'R': 5, 'O': 6, 'J': 7, 'C': 8, 'G': 9, 'U': 10, 'N': 11, 'B': 12, 'K': 13, 'F': 14, 'A': 15, 'L': 16, 'I': 17, 'S': 18, 'V': 19, 'D': 20, 'M': 21, 'Y': 22, '.': 23, 'W': 24, ',': 25, '-': 26, '2': 27, '0': 28, '6': 29, '1': 30, '5': 31, '9': 32, '7': 33, '3': 34, '4': 35, 'X': 36, 'Z': 37, '8': 38, 'Q': 39, '?': 40}


def clean_text(text):
    '''Remove unwanted characters and extra spaces from the text'''
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]', '', text)
    text = re.sub('a0', '', text)
    text = re.sub('\'92t', '\'t', text)
    text = re.sub('\'92s', '\'s', text)
    text = re.sub('\'92m', '\'m', text)
    text = re.sub('\'92ll', '\'ll', text)
    text = re.sub('\'91', '', text)
    text = re.sub('\'92', '', text)
    text = re.sub('\'93', '', text)
    text = re.sub('\'94', '', text)
    text = re.sub('\.', '. ', text)
    text = re.sub('\!', '! ', text)
    text = re.sub('\?', '? ', text)
    text = re.sub(' +', ' ', text)
    return text


# Add special tokens to vocab_to_int
vocab_to_int = {'T': 0, 'h': 1, 'e': 2, ' ': 3, 'P': 4, 'r': 5, 'o': 6, 'j': 7, 'c': 8, 't': 9, 'G': 10, 'u': 11, 'n': 12, 'b': 13, 'g': 14, 'E': 15, 'B': 16, 'k': 17, 'f': 18, 'y': 19, 'd': 20, 'a': 21, 's': 22, 'l': 23, 'i': 24, '\r': 25, 'A': 26, 'R': 27, 'N': 28, 'C': 29, ',': 30, 'S': 31, 'U': 32, 'w': 33, 'm': 34, 'v': 35, '.': 36, 'Y': 37, 'p': 38, '-': 39, 'L': 40, ':': 41, 'D': 42, 'J': 43, '1': 44, '2': 45, '0': 46, '6': 47, '8': 48, '7': 49, '3': 50, 'I': 51, 'O': 52, '5': 53, '9': 54, 'F': 55, 'H': 56, 'K': 57, 'M': 58, '/': 59, '�': 60, 'q': 61, 'Q': 62, 'Z': 63, 'V': 64, 'x': 65, 'z': 66, ';': 67, '&': 68, '4': 69, '?': 70, '!': 71, '"': 72, 'X': 73, 'W': 74, "'": 75, '$': 76, '<': 77, '|': 78, '~': 79, '\t': 80}
count = 0


# Add special tokens to vocab_to_int
print (vocab_to_int);
codes = ['<PAD>', '<EOS>', '<GO>']
for code in codes:
    vocab_to_int[code] = count
    count += 1

'''int_to_vocab = {}
for character, value in vocab_to_int.items():
    int_to_vocab[value] = character
'''
int_to_vocab = {0: 'T', 1: 'H', 2: 'E', 3: ' ', 4: 'P', 5: 'R', 6: 'O', 7: 'J', 8: 'C', 9: 'G', 10: 'U', 11: 'N', 12: 'B', 13: 'K', 14: 'F', 15: 'A', 16: 'L', 17: 'I', 18: 'S', 19: 'V', 20: 'D', 21: 'M', 22: 'Y', 23: '.', 24: 'W', 25: ',', 26: '-', 27: '2', 28: '0', 29: '6', 30: '1', 31: '5', 32: '9', 33: '7', 34: '3', 35: '4', 36: 'X', 37: 'Z', 38: '8', 39: 'Q', 40: '?', 41: '<PAD>', 42: '<EOS>', 43: '<GO>'}
print (int_to_vocab)


int_to_vocab = {}
for character, value in vocab_to_int.items():
    int_to_vocab[value] = character




def text_to_ints(text):
    '''Prepare the text for the model'''
    text = clean_text(text)
    return [vocab_to_int[word] for word in text]

epochs = 100
batch_size = 128
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.95
keep_probability = 0.75
# In[176]:

# Create your own sentence or use one from the dataset
text = "COL CUAUQHTEMOC"
text = text_to_ints(text)

# random = np.random.randint(0,len(testing_sorted))
# text = testing_sorted[random]
# text = noise_maker(text, 0.95)

checkpoint = "./kp=0.75,nl=2,th=0.95.ckpt"

model = tf.train.import_meta_graph('kp=0.75,nl=2,th=0.95.ckpt.meta')
config = tf.ConfigProto()
graph = tf.get_default_graph()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
with tf.Session(config = config, graph = graph) as sess:
    # Load saved model
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint))#tf.train.Saver()
    #saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    grafo = tf.get_default_graph()

    print("Model restored")


    # Multiply by batch_size to match the model's input parameters
    #inputs = graph.get_operation_by_name("input").outputs[0]
    #prediction = graph.get_operation_by_name("prediction").outputs[0]

    answer_logits = sess.run(grafo.get_tensor_by_name("predictions/predictions:0"), {grafo.get_tensor_by_name("inputs/inputs:0"): [text] * batch_size,
                                                 grafo.get_tensor_by_name("inputs_length:0"): [len(text)] * batch_size,
                                                 grafo.get_tensor_by_name("targets_length:0"): [len(text) + 1],
                                                 grafo.get_tensor_by_name("keep_prob:0"): [1.0]})[0]

# Remove the padding from the generated sentence
pad = vocab_to_int["<PAD>"]

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))


text = "FRACC REEOS TECAMAC SEC BOGSQUES."

text = text_to_ints(text)

# random = np.random.randint(0,len(testing_sorted))
# text = testing_sorted[random]
# text = noise_maker(text, 0.95)

checkpoint = "./kp=0.75,nl=2,th=0.95.ckpt"

model = tf.train.import_meta_graph('kp=0.75,nl=2,th=0.95.ckpt.meta')
config = tf.ConfigProto()
graph = tf.get_default_graph()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
with tf.Session(config = config, graph = graph) as sess:
    # Load saved model
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint))#tf.train.Saver()
    #saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    grafo = tf.get_default_graph()

    print("Model restored")


    # Multiply by batch_size to match the model's input parameters
    #inputs = graph.get_operation_by_name("input").outputs[0]
    #prediction = graph.get_operation_by_name("prediction").outputs[0]

    answer_logits = sess.run(grafo.get_tensor_by_name("predictions/predictions:0"), {grafo.get_tensor_by_name("inputs/inputs:0"): [text] * batch_size,
                                                 grafo.get_tensor_by_name("inputs_length:0"): [len(text)] * batch_size,
                                                 grafo.get_tensor_by_name("targets_length:0"): [len(text) + 1],
                                                 grafo.get_tensor_by_name("keep_prob:0"): [1.0]})[0]

# Remove the padding from the generated sentence

pad = vocab_to_int["<PAD>"]

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))


# Examples of corrected sentences:
# - Spellin is difficult, whch is wyh you need to study everyday.
# - Spelling is difficult, which is why you need to study everyday.
#
#
# - The first days of her existence in th country were vrey hard for Dolly.
# - The first days of her existence in the country were very hard for Dolly.
#
#
# - Thi is really something impressiv thaat we should look into right away!
# - This is really something impressive that we should look into right away!

# ## Summary

# I hope that you have found this project to be rather interesting and useful. The example sentences that I have presented above were specifically chosen, and the model will not always be able to make corrections of this quality. Given the amount of data that we are working with, this model still struggles. For it to be more useful, it would require far more training data, and additional parameter tuning. This parameter values that I have above worked best for me, but I expect there are even better values that I was not able to find.
#
# Thanks for reading!