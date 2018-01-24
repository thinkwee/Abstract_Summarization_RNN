import tensorflow as tf
from w2v import w2v
import logging
import logging.config
import string
import numpy as np
import pickle
from seq2seq import seq2seqmodel

LOG_FILE = './log/test.log'
handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
# fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
# formatter = logging.Formatter(fmt)  # 实例化formatter
# handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('testlogger')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)


def build_embed_matrix():
    # vocab_size
    # embed_size
    # batch_size
    # num_sampled
    # learning_rate
    # skip_windows
    # data_name
    # num_train_steps
    # skip_steps
    w2vmodel = w2v(18500, 128, 32, 64, 1.0, 1, 'testtext.zip', 17000, 100)
    w2vmodel.build_graph()
    return w2vmodel.train()


def load_embed_matrix():
    pkl_file = open('./save/embed_matrix.pkl', 'rb')
    embed_matrix = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('./save/one_hot_dictionary.pkl', 'rb')
    one_hot_dictionary = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('./save/one_hot_dictionary_index.pkl', 'rb')
    one_hot_dictionary_index = pickle.load(pkl_file)
    pkl_file.close()
    return embed_matrix, one_hot_dictionary, one_hot_dictionary_index


def get_batch(batch_size, iterator):
    while True:
        encoder_batch = []
        encoder_length_batch = np.zeros([batch_size], dtype=int)

        for index in range(batch_size):
            encoder_input_batch_single, encoder_length_single = next(iterator)
            encoder_batch.append(encoder_input_batch_single)
            encoder_length_batch[index] = encoder_length_single

        yield encoder_batch, encoder_length_batch


def one_hot_generate(one_hot_dictionary):
    file_article = open('./data/test.txt', 'rb')
    sentence_article = bytes.decode(file_article.readline())

    while sentence_article:
        words_article = []
        count_article = 0
        strip = string.whitespace + string.punctuation + "\"'"
        for word in sentence_article.split():
            word = word.strip(strip)
            words_article.append(word)
            count_article += 1

        one_hot_article = np.zeros([70], dtype=int)

        for index, word in enumerate(words_article):
            one_hot_article[index] = one_hot_dictionary[word] if word in one_hot_dictionary else 0

        yield one_hot_article, count_article,
        sentence_article = bytes.decode(file_article.readline())

    file_article.close()


def main():
    embed_matrix, one_hot_dictionary, one_hot_dictionary_index = load_embed_matrix()
    logger.debug("w2v restored")

    seq2seq_basic_rnn_without_attention = seq2seqmodel(18500, 128, 128, 128, 1, embed_matrix)
    seq2seq_basic_rnn_without_attention._build_graph()
    logger.debug("seq2seq model built")

    single_generate = one_hot_generate(one_hot_dictionary)
    batches = get_batch(1, single_generate)
    logger.debug("batch generated")

    seq2seq_basic_rnn_without_attention._test(1, batches, one_hot_dictionary_index)
    logger.debug("seq2seq model tested")


if __name__ == '__main__':
    main()
