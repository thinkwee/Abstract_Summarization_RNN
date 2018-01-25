import tensorflow as tf
from w2v import w2v
import logging
import logging.config
import string
import numpy as np
import pickle
from seq2seq import seq2seqmodel

LOG_FILE = './log/train.log'
handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
# fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
# formatter = logging.Formatter(fmt)  # 实例化formatter
# handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('trainlogger')  # 获取名为tst的logger
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


def save_embed_matrix(embed_matrix, one_hot_dictionary, one_hot_dictionary_index):
    output = open('./save/embed_matrix.pkl', 'wb')
    pickle.dump(embed_matrix, output)
    output.close()
    output = open('./save/one_hot_dictionary.pkl', 'wb')
    pickle.dump(one_hot_dictionary, output)
    output.close()
    output = open('./save/one_hot_dictionary_index.pkl', 'wb')
    pickle.dump(one_hot_dictionary_index, output)
    output.close()


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
        decoder_batch = []
        target_batch = []
        encoder_length_batch = np.zeros([batch_size], dtype=int)
        decoder_length_batch = np.zeros([batch_size], dtype=int)

        for index in range(batch_size):
            encoder_input_batch_single, decoder_input_batch_single, target_batch_single, encoder_length_single, decoder_length_single = next(
                iterator)
            encoder_batch.append(encoder_input_batch_single)
            decoder_batch.append(decoder_input_batch_single)
            target_batch.append(target_batch_single)
            encoder_length_batch[index] = encoder_length_single
            decoder_length_batch[index] = decoder_length_single

        yield encoder_batch, decoder_batch, target_batch, encoder_length_batch, decoder_length_batch


def one_hot_generate(one_hot_dictionary, epoch):
    for i in range(epoch):
        file_article = open('./data/article.txt', 'rb')
        sentence_article = bytes.decode(file_article.readline())
        file_headline = open('./data/headline.txt', 'rb')
        sentence_headline = bytes.decode(file_headline.readline())

        while sentence_article and sentence_headline:
            words_article = []
            words_headline = []
            count_article = 0
            count_headline = 0
            strip = string.whitespace + string.punctuation + "\"'"
            for word in sentence_article.split():
                word = word.strip(strip)
                words_article.append(word)
                count_article += 1
            for word in sentence_headline.split():
                word = word.strip(strip)
                words_headline.append(word)
                count_headline += 1
            one_hot_article = np.zeros([70], dtype=int)
            one_hot_headline_input = np.zeros([26], dtype=int)
            one_hot_headline_target = np.zeros([26], dtype=int)

            for index, word in enumerate(words_article):
                one_hot_article[index] = one_hot_dictionary[word] if word in one_hot_dictionary else 0

            one_hot_headline_input[0] = 0
            for index, word in enumerate(words_headline):
                one_hot_headline_input[index + 1] = one_hot_dictionary[word] if word in one_hot_dictionary else 0
                one_hot_headline_target[index] = one_hot_dictionary[word] if word in one_hot_dictionary else 0

            yield one_hot_article, one_hot_headline_input, one_hot_headline_target, count_article, count_headline
            sentence_article = bytes.decode(file_article.readline())
            sentence_headline = bytes.decode(file_headline.readline())

        file_headline.close()
        file_article.close()


def main():
    # embed_matrix, one_hot_dictionary, one_hot_dictionary_index = build_embed_matrix()
    # logger.debug("w2v finished")
    #
    # save_embed_matrix(embed_matrix, one_hot_dictionary, one_hot_dictionary_index)
    # logger.debug("w2v saved")

    embed_matrix, one_hot_dictionary, one_hot_dictionary_index = load_embed_matrix()
    logger.debug("w2v restored")

    seq2seq_basic_rnn_without_attention = seq2seqmodel(18500, 128, 128, 128, 32, embed_matrix)
    seq2seq_basic_rnn_without_attention._build_graph()
    logger.debug("seq2seq model built")

    single_generate = one_hot_generate(one_hot_dictionary, 100)
    batches = get_batch(32, single_generate)
    logger.debug("batch generated")

    seq2seq_basic_rnn_without_attention._train(100, 220, batches, 20)
    logger.debug("seq2seq model trained")


if __name__ == '__main__':
    main()
