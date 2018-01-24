import tensorflow as tf
import logging.config
import tensorflow.contrib.seq2seq as s2s
from tensorflow.contrib.layers import *

LOG_FILE = './log/seq2seq.log'
MODEL_FILE = './model/'
handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('seq2seqlogger')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)


class seq2seqmodel:
    def __init__(self, vocab_size, embed_size, encoder_hidden_units, decoder_hidden_units, batch_size,
                 embed_matrix_init):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        self.batch_size = batch_size
        self.embed_matrix_init = embed_matrix_init

    def _create_placeholder(self):
        with tf.name_scope("data_seq2seq"):
            self.encoder_inputs = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='encoder_inputs')
            self.decoder_inputs = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='decoder_inputs')
            self.decoder_targets = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='decoder_targets')
            self.encoder_length = tf.placeholder(shape=(self.batch_size), dtype=tf.int32, name='encoder_length')
            self.decoder_length = tf.placeholder(shape=(self.batch_size), dtype=tf.int32, name='decoder_length')

    def _create_embedding(self):
        self.embeddings_trainable = tf.Variable(initial_value=self.embed_matrix_init, name='word_embedding_train')
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings_trainable, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings_trainable, self.decoder_inputs)

    def _create_cell(self):
        """output: all the hidden state output
         final_state:the last hidden state output
         In seq2seq model we just need the encoder's final state as the initial state of decoder.On the contrary we just use output of the
        decoder to predict the output word """
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            encoder_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.encoder_hidden_units)
            self.encoder_output, self.encoder_final_state = tf.nn.dynamic_rnn(
                cell=encoder_cell,
                inputs=self.encoder_inputs_embedded,
                dtype=tf.float32,
                sequence_length=self.encoder_length,
                time_major=False
            )
        with tf.variable_scope('decoder_train', reuse=tf.AUTO_REUSE):
            decoder_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.decoder_hidden_units)
            self.decoder_outputs_train, _ = tf.nn.dynamic_rnn(
                cell=decoder_cell,
                inputs=self.decoder_inputs_embedded,
                initial_state=self.encoder_final_state,
                dtype=tf.float32,
                sequence_length=self.decoder_length,
                time_major=False
            )
        with tf.variable_scope('decoder_test', reuse=tf.AUTO_REUSE):
            start_tokens = 0
            end_tokens = 0

            # Helper
            helper = s2s.GreedyEmbeddingHelper(
                self.embeddings_trainable,
                tf.fill([self.batch_size], start_tokens), end_tokens)
            # Decoder
            decoder = s2s.BasicDecoder(
                decoder_cell, helper, self.encoder_final_state)
            # Dynamic decoding
            outputs, _, _ = s2s.dynamic_decode(
                decoder, maximum_iterations=25)
            self.decoder_prediction = outputs.sample_id

    def _create_loss(self):
        """use linear layer to project the output of decoder to predict the ouput word"""
        with tf.name_scope("loss_seq2seq"):
            self.decoder_logits = tf.contrib.layers.linear(self.decoder_outputs_train, self.vocab_size)
            # self.decoder_logits_test = tf.contrib.layers.linear(self.decoder_outputs_test, self.vocab_size)
            # self.decoder_prediction = tf.argmax(self.decoder_logits_test, 2)
            self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
                logits=self.decoder_logits,
            )
            self.loss = tf.reduce_mean(self.stepwise_cross_entropy)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope("summaries_seq2seq"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def _build_graph(self):
        self._create_placeholder()
        self._create_embedding()
        self._create_cell()
        self._create_loss()
        self._create_summaries()
        print("seq2seq graph built")

    def _train(self, epoch, num_train_steps, batches, skip_steps):
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.total_loss = 0.0
            print("start training seq2seq model")
            writer = tf.summary.FileWriter('./graphs/seq2seq', sess.graph)
            for i in range(epoch):
                for index in range(num_train_steps):
                    print("epoch: %d at train step: %d" % (i, index + 1))
                    encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length = next(batches)
                    feed_dict = {
                        self.decoder_targets: decoder_targets,
                        self.encoder_length: encoder_length,
                        self.decoder_length: decoder_length,
                        self.encoder_inputs: encoder_inputs,
                        self.decoder_inputs: decoder_inputs,
                    }
                    loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op],
                                                      feed_dict=feed_dict)
                    self.total_loss += loss_batch
                    writer.add_summary(summary, global_step=index)
                    if (index + 1) % skip_steps == 0:
                        logger.debug('seq2seq average loss at epoch {} step {} : {:5.1f}'.format(i, index + 1,
                                                                                                 self.total_loss / skip_steps))
                        self.total_loss = 0.0
            saver.save(sess, MODEL_FILE + 'model.ckpt', global_step=num_train_steps)
            logger.debug("seq2seq trained,model saved")

    def _test(self, num_train_steps, batches, one_hot_dictionary_index):
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(MODEL_FILE)
        with tf.Session() as sess:
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("the model has been successfully restored")
                sess.run(tf.global_variables_initializer())
                for index in range(num_train_steps):
                    train_inputs, encoder_length = next(batches)
                    feed_dict = {self.encoder_inputs: train_inputs,
                                 self.encoder_length: encoder_length
                                 }
                    prediction = sess.run(self.decoder_prediction, feed_dict=feed_dict)
                    prediction = prediction[0]
                    print(prediction)
                    answer = [one_hot_dictionary_index[prediction[i]] for i in range(len(prediction))]
                    print("test answer: ")
                    print(answer)
            else:
                print("model restored failed")
                pass
