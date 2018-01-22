from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from process_data import process_data
import logging
import logging.config

# VOCAB_SIZE = 16800
# BATCH_SIZE = 32
# EMBED_SIZE = 128
# SKIP_WINDOW = 1
# NUM_SAMPLED = 64
# LEARNING_RATE = 1.0
# NUM_TRAIN_STEPS = 13000
# SKIP_STEP = 100
# DATA_NAME = 'article.zip'

LOG_FILE = './log/w2v.log'
handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('w2vlogger')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)


class w2v:

    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate, skip_windows, data_name,
                 num_train_steps, skip_steps):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.win = skip_windows
        self.data_name = data_name
        self.num_train_steps = num_train_steps
        self.skip_steps = skip_steps
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        with tf.name_scope('data'):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_words')
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')
        with tf.name_scope('embedding_matrix'):
            self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0),
                                            name='embed_matrix')

    def _create_loss(self):
        with tf.name_scope('loss'):
            self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')

            nce_weight = tf.Variable(
                tf.truncated_normal([self.vocab_size, self.embed_size], stddev=1.0 / (self.embed_size ** 0.5)),
                name='nce_weigh')

            nce_bias = tf.Variable(tf.zeros([self.vocab_size]), name='nce_bias')

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                      biases=nce_bias,
                                                      labels=self.target_words,
                                                      inputs=self.embed,
                                                      num_sampled=self.num_sampled,
                                                      num_classes=self.vocab_size), name='loss')

    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        logger.debug('w2v graph for %s has been build', self.data_name)

    def train(self):
        print("start w2v train for %s" % self.data_name)
        batch_gen, one_hot_dictionary = process_data(self.vocab_size, self.batch_size, self.win, self.data_name)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_loss = 0.0
            writer = tf.summary.FileWriter('./graphs/w2v/', sess.graph)
            for index in range(self.num_train_steps):
                centers, targets = next(batch_gen)
                loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op],
                                                  feed_dict={self.center_words: centers, self.target_words: targets})
                total_loss += loss_batch
                writer.add_summary(summary, global_step=index)
                if (index + 1) % self.skip_steps == 0:
                    logger.debug('w2v for {} average loss at step {} : {:5.1f}'.format(self.data_name, index + 1,
                                                                                       total_loss / self.skip_steps))
                    total_loss = 0.0
                writer.close()
            self.final_embed_matrix = sess.run(self.embed_matrix)
        logger.debug('w2v train for %s has finished', self.data_name)
        print('embed_matrix for %s has been build' % self.data_name)
        return self.final_embed_matrix, one_hot_dictionary
