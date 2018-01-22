import tensorflow as tf
from w2v import w2v
import logging
import logging.config
import string
import numpy
import pickle

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
    modelheadline = w2v(18500, 128, 32, 64, 1.0, 1, 'traintext.zip', 17000, 100)
    modelheadline.build_graph()
    return modelheadline.train()


def save_embed_matrix(embed_matrix_headline, one_hot_dictionary_headline):
    output = open('./save/embed_matrix_headline.pkl', 'wb')
    pickle.dump(embed_matrix_headline, output)
    output.close()
    output = open('./save/one_hot_dictionary_headline.pkl', 'wb')
    pickle.dump(one_hot_dictionary_headline, output)
    output.close()


def load_embed_matrix():
    pkl_file = open('./save/embed_matrix_headline.pkl', 'rb')
    embed_matrix_headline = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open('./save/one_hot_dictionary_headline.pkl', 'rb')
    one_hot_dictionary_headline = pickle.load(pkl_file)
    pkl_file.close()
    return embed_matrix_headline, one_hot_dictionary_headline


def test(embed_matrix_headline, one_hot_dictionary_headline):
    sentence = "the british economy is poised for strong growth into 2005 raising the possibility of an interest rate " \
               "hike early in the new year according to a study published monday "
    words = []
    strip = string.whitespace + string.punctuation + string.digits + "\"'"
    for word in sentence.split():
        word = word.strip(strip)
        words.append(word)
    print(words)

    onehotindex = [one_hot_dictionary_headline[word] if word in one_hot_dictionary_headline else 0 for word in words]

    print(onehotindex)

    with tf.name_scope("embed"):
        embed = tf.nn.embedding_lookup(embed_matrix_headline, onehotindex, name='embed')

    with tf.Session() as sess:
        embed = sess.run(embed)
        for i in range(len(words)):
            for j in range(128):
                logger.debug("%f", embed[i][j])


def main():
    # embed_matrix_headline, one_hot_dictionary_headline = build_embed_matrix()
    # save_embed_matrix(embed_matrix_headline, one_hot_dictionary_headline)
    embed_matrix_headline, one_hot_dictionary_headline = load_embed_matrix()
    test(embed_matrix_headline, one_hot_dictionary_headline)


if __name__ == '__main__':
    main()
