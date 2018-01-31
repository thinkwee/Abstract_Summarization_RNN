import tensorflow as tf
from w2v import w2v
import logging
import logging.config
import string
import numpy as np
import pickle
from seq2seq import seq2seqmodel
from pyrouge import Rouge155

LOG_FILE = './log/test.log'
handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
# fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
# formatter = logging.Formatter(fmt)  # 实例化formatter
# handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('testlogger')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)

"""Hyper Parameters"""
VOVAB_SIZE = 18500
EMBED_SIZE = 128
ENCODER_HIDEEN_UNITS = 128
DECODER_HIDDEN_UNITS = 128
BATCH_SIZE = 1
ENCODER_LAYERS = 1
EPOCH = 1
NUM_TRAIN_STEPS = 1
SKIP_STEPS = 20
LEARNING_RATE = 0.1


def RunPaperMyRougeHtml(system_path, model_path, modelpattern, systempattern, config_file_path=None,
                        perlpath=r'/usr/local/lib/x86_64-linux-gnu/perl/5.22.1/bin/perl', system_idstr=['None']):
    r = Rouge155()
    r.system_dir = system_path
    r.config_file = config_file_path
    r.model_dir = model_path
    r.system_filename_pattern = systempattern
    r.model_filename_pattern = modelpattern
    output = r.evaluate(system_id=system_idstr, conf_path=config_file_path, PerlPath=perlpath)
    print(output)
    return output


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
            one_hot_headline_input = np.zeros([30], dtype=int)
            one_hot_headline_target = np.zeros([30], dtype=int)

            for index, word in enumerate(words_article):
                one_hot_article[index] = one_hot_dictionary[word] if word in one_hot_dictionary else 0

            one_hot_headline_input[0] = 18498
            one_hot_headline_target[0] = 18498
            for index, word in enumerate(words_headline):
                one_hot_headline_input[index + 1] = one_hot_dictionary[word] if word in one_hot_dictionary else 0
                one_hot_headline_target[index + 1] = one_hot_dictionary[word] if word in one_hot_dictionary else 0
            yield one_hot_article, one_hot_headline_input, one_hot_headline_target, count_article, count_headline
            sentence_article = bytes.decode(file_article.readline())
            sentence_headline = bytes.decode(file_headline.readline())

        file_headline.close()
        file_article.close()


def main():
    embed_matrix, one_hot_dictionary, one_hot_dictionary_index = load_embed_matrix()
    logger.debug("w2v restored")

    seq2seq_basic_rnn_without_attention = seq2seqmodel(vocab_size=VOVAB_SIZE,
                                                       embed_size=EMBED_SIZE,
                                                       encoder_hidden_units=ENCODER_HIDEEN_UNITS,
                                                       decoder_hidden_units=DECODER_HIDDEN_UNITS,
                                                       encoder_layers=ENCODER_LAYERS,
                                                       batch_size=BATCH_SIZE,
                                                       learning_rate=LEARNING_RATE,
                                                       embed_matrix_init=embed_matrix)
    seq2seq_basic_rnn_without_attention._build_graph()
    logger.debug("seq2seq model built")

    single_generate = one_hot_generate(one_hot_dictionary,
                                       epoch=1)
    batches = get_batch(batch_size=BATCH_SIZE,
                        iterator=single_generate)
    logger.debug("batch generated")

    seq2seq_basic_rnn_without_attention._test(num_train_steps=NUM_TRAIN_STEPS,
                                              batches=batches,
                                              one_hot_dictionary_index=one_hot_dictionary_index)
    logger.debug("seq2seq model tested")


if __name__ == '__main__':
    main()
