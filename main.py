import tensorflow as tf
from w2v import w2v
import logging
import logging.config

LOG_FILE = 'main.log'
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)  # 实例化handler
# fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
# formatter = logging.Formatter(fmt)  # 实例化formatter
# handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('mainlogger')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)


def main():
    # def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate, skip_windows, data_name,
    #              num_train_steps, skip_steps):

    modelheadline = w2v(10200, 128, 32, 64, 1.0, 1, 'headline.zip', 4400, 10)
    modelheadline.build_graph()
    embed_matrix_headline, one_hot_dictionary_headline = modelheadline.train()
    word = "kingdom"
    onehotindex = one_hot_dictionary_headline[word]
    print(onehotindex)

    with tf.name_scope("embed"):
        embed = tf.nn.embedding_lookup(embed_matrix_headline, onehotindex, name='embed')

    with tf.Session() as sess:
        embed = sess.run(embed)
        for i in range(128):
            logger.debug("%f", embed[i])

    # print(embed_matrix_headline)

    # modelarticle = w2v(16800, 128, 32, 64, 1.0, 1, 'article.zip', 13000, 100)
    # modelarticle.build_graph()
    # embed_matrix_article = modelarticle.train()


if __name__ == '__main__':
    main()
