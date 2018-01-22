import tensorflow as tf
import logging.config

LOG_FILE = './log/seq2seq.log'
handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter
logger = logging.getLogger('seq2seqlogger')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)


class seq2seqmodel:
    def __init__(self, vocab_size, embed_size, encoder_hidden_units, decoder_hidden_units, batch_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        self.batch_size = batch_size

    def _create_placeholder(self):
        with tf.name_scope("data"):
            self.encoder_inputs = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='encoder_inputs')
            self.decoder_inputs = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='decoder_inputs')
            self.decoder_targets = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='decoder_targets')
        with tf.name_scope("word_embedding"):
            self.embeddings = tf.placeholder(shape=(self.vocab_size, self.embed_size), dtype=tf.float32,
                                             name='embeddings')

    def _create_embedding(self):
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

    def _create_cell(self):
        """output: all the hidden state output
         final_state:the last hidden state output
         In seq2seq model we just need the encoder's final state as the initial state of decoder.On the contrary we just use output of the
        decoder to predict the output word """
        # self.seq_length = [120 for i in range(self.batch_size)]
        with tf.variable_scope('encoder_cell', reuse=tf.AUTO_REUSE):
            encoder_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.encoder_hidden_units)
            self.encoder_output, self.encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell, self.encoder_inputs_embedded,
                dtype=tf.float32, time_major=False
            )
        print(self.encoder_final_state)
        print(self.encoder_output)
        with tf.variable_scope('decoder_cell', reuse=tf.AUTO_REUSE):
            decoder_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.decoder_hidden_units)
            self.decoder_outputs, _ = tf.nn.dynamic_rnn(
                decoder_cell, self.decoder_inputs_embedded,
                initial_state=self.encoder_final_state,
                dtype=tf.float32, time_major=False
            )

    def _create_loss(self):
        """use linear layer to project the output of decoder to predict the ouput word"""
        with tf.name_scope("loss"):
            self.decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.vocab_size)
            self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
            self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
                logits=self.decoder_logits,
            )
            self.loss = tf.reduce_mean(self.stepwise_cross_entropy)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def _build_graph(self):
        self._create_placeholder()
        self._create_embedding()
        self._create_cell()
        self._create_loss()
        self._create_summaries()
        print("seq2seq graph built")

    def _train(self, num_train_steps, batches, skip_steps, embed_matrix):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.total_loss = 0.0
            print("start training seq2seq model")
            writer = tf.summary.FileWriter('./graphs/seq2seq', sess.graph)
            for index in range(num_train_steps):
                print("train step: %d" % (index + 1))
                encoder_inputs, decoder_inputs = next(batches)
                decoder_targets = decoder_inputs
                # encoder_inputs_, _ = make_batch(batch)
                # decoder_targets_, _ = make_batch([(sequence) + [EOS] for sequence in batch])
                # decoder_inputs_, _ = make_batch([[EOS] + (sequence) for sequence in batch])
                feed_dict = {self.encoder_inputs: encoder_inputs,
                             self.decoder_inputs: decoder_inputs,
                             self.decoder_targets: decoder_targets,
                             self.embeddings: embed_matrix
                             }
                loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op],
                                                  feed_dict=feed_dict)
                self.total_loss += loss_batch
                writer.add_summary(summary, global_step=index)
                if (index + 1) % skip_steps == 0:
                    logger.debug('seq2seq average loss at step {} : {:5.1f}'.format(index + 1,
                                                                                    self.total_loss / skip_steps))
                    self.total_loss = 0.0
                    # predict_ = sess.run(self.decoder_prediction, feed_dict)
