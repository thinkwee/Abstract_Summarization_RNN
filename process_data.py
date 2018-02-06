from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import random
import sys

import zipfile

import numpy as np
import tensorflow as tf

import utils
import logging
import logging.config

sys.path.append('..')

DATA_FOLDER = 'data/'
LOG_FILE = './log/processdata.log'

handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('datalogger')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)


def read_data(file_path):
    logger.debug('read data')
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
        # tf.compat.as_str() converts the input into the string
    return words


def build_vocab(words, vocab_size):
    logger.debug('build vocabulary')
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    """ Save first 1000 words in vocab_1000 for projection """
    dictionary = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    utils.make_dir('processed')
    with open('processed/vocab_1000.tsv', "w") as f:
        for word, _ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary


def convert_words_to_index(words, dictionary):
    logger.debug('convert')
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]


def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    count = 0
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target


def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch


def process_data(vocab_size, batch_size, skip_window, data_name):
    logger.debug('process data')
    file_path = DATA_FOLDER + data_name
    words = read_data(file_path)
    dictionary, index_dictionary = build_vocab(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    del words  # to save memory
    single_gen = generate_sample(index_words, skip_window)
    return get_batch(single_gen, batch_size), dictionary, index_dictionary


def get_index_vocab(vocab_size, data_name):
    file_path = DATA_FOLDER + data_name
    words = read_data(file_path)
    return build_vocab(words, vocab_size)
