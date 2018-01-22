import tensorflow as tf
from w2v import w2v
import logging
import logging.config
import string
import numpy as np
import pickle
from seq2seq import seq2seqmodel

LOG_FILE = './log/main.log'
handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
# fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
# formatter = logging.Formatter(fmt)  # 实例化formatter
# handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('mainlogger')  # 获取名为tst的logger
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
    w2vmodel = w2v(18500, 128, 32, 64, 1.0, 1, 'traintext.zip', 17000, 100)
    w2vmodel.build_graph()
    return w2vmodel.train()


def save_embed_matrix(embed_matrix, one_hot_dictionary):
    output = open('./save/embed_matrix.pkl', 'wb')
    pickle.dump(embed_matrix, output)
    output.close()
    output = open('./save/one_hot_dictionary.pkl', 'wb')
    pickle.dump(one_hot_dictionary, output)
    output.close()


def load_embed_matrix():
    pkl_file = open('./save/embed_matrix.pkl', 'rb')
    embed_matrix = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('./save/one_hot_dictionary.pkl', 'rb')
    one_hot_dictionary = pickle.load(pkl_file)
    pkl_file.close()
    return embed_matrix, one_hot_dictionary


def get_batch(batch_size, vocab_size, iterator):
    while True:
        input_batch = []
        target_batch = []
        for index in range(batch_size):
            input_batch_single, target_batch_single = next(iterator)
            input_batch.append(input_batch_single)
            target_batch.append(target_batch_single)
        yield input_batch, target_batch


def one_hot_generate(one_hot_dictionary):
    file_article = open('./data/article.txt', 'rb')
    sentence_article = bytes.decode(file_article.readline())
    file_headline = open('./data/headline.txt', 'rb')
    sentence_headline = bytes.decode(file_headline.readline())

    while sentence_article and sentence_headline:
        words_article = []
        words_headline = []
        strip = string.whitespace + string.punctuation + "\"'"
        for word in sentence_article.split():
            word = word.strip(strip)
            words_article.append(word)
        for word in sentence_headline.split():
            word = word.strip(strip)
            words_headline.append(word)
        one_hot_headline = np.zeros([25], dtype=int)
        one_hot_article = np.zeros([70], dtype=int)

        for index, word in enumerate(words_headline):
            one_hot_headline[index] = one_hot_dictionary[word] if word in one_hot_dictionary else 0

        for index, word in enumerate(words_article):
            one_hot_article[index] = one_hot_dictionary[word] if word in one_hot_dictionary else 0

        # print(one_hot_article)

        yield one_hot_article, one_hot_headline
        sentence_article = bytes.decode(file_article.readline())
        sentence_headline = bytes.decode(file_headline.readline())

    file_headline.close()
    file_article.close()


def main():
    # embed_matrix, one_hot_dictionary = build_embed_matrix()
    # save_embed_matrix(embed_matrix, one_hot_dictionary)
    embed_matrix, one_hot_dictionary = load_embed_matrix()
    seq2seq_basic_rnn_without_attention = seq2seqmodel(18500, 128, 128, 128, 32)
    single_generate = one_hot_generate(one_hot_dictionary)
    batches = get_batch(32, 18500, single_generate)
    seq2seq_basic_rnn_without_attention._build_graph()
    seq2seq_basic_rnn_without_attention._train(220, batches, 10, embed_matrix)


if __name__ == '__main__':
    main()
