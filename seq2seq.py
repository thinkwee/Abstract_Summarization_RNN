import tensorflow as tf
import logging.config
import tensorflow.contrib.seq2seq as s2s
import tensorflow.contrib as contrib
import tensorflow.contrib.rnn as rnn
import numpy as np


class seq2seqmodel:
    def __init__(self, vocab_size, embed_size, encoder_hidden_units, decoder_hidden_units, batch_size,
                 embed_matrix_init, encoder_layers, learning_rate, is_train):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        self.batch_size = batch_size
        self.embed_matrix_init = embed_matrix_init
        self.encoder_layers = encoder_layers
        self.learning_rate = learning_rate
        self.is_train = is_train
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.MODEL_FILE = './model/'

    def _create_placeholder(self):
        with tf.name_scope("data_seq2seq"):
            self.encoder_inputs = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='encoder_inputs')
            self.decoder_inputs = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='decoder_inputs')
            self.decoder_targets = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='decoder_targets')
            self.decoder_length = tf.placeholder(shape=self.batch_size, dtype=tf.int32, name='decoder_length')

    def _create_embedding(self):
        self.embeddings_trainable = tf.Variable(initial_value=self.embed_matrix_init, name='word_embedding_train')
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings_trainable, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings_trainable, self.decoder_inputs)

    def _create_seq2seq(self):

        # multi layer blstm encoder
        for layer_i in range(self.encoder_layers):
            with tf.variable_scope('encoder%i' % layer_i, reuse=tf.AUTO_REUSE):
                cell_fw = rnn.LSTMCell(
                    num_units=self.encoder_hidden_units,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=114),
                    state_is_tuple=True)
                cell_bw = rnn.LSTMCell(
                    num_units=self.encoder_hidden_units,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=133),
                    state_is_tuple=True)
                (self.encoder_inputs_embedded, self.encoder_final_state) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=self.encoder_inputs_embedded,
                    dtype=tf.float32)
        self.encoder_final_state_c = tf.concat(
            (self.encoder_final_state[0].c, self.encoder_final_state[1].c), 1)
        self.encoder_final_state_h = tf.concat(
            (self.encoder_final_state[0].h, self.encoder_final_state[1].h), 1)
        self.encoder_final_state = contrib.rnn.LSTMStateTuple(
            c=self.encoder_final_state_c,
            h=self.encoder_final_state_h)

        # Basic Lstm Decoder for train and infer
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            self.decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.decoder_hidden_units,
                                                        name='decoder_cell',
                                                        state_is_tuple=True)
            self.fc_layer = tf.layers.Dense(self.vocab_size, name='dense_layer')

            # for train
            self.helper_train = contrib.seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                               sequence_length=self.decoder_length)
            self.decoder_train = contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                              initial_state=self.encoder_final_state,
                                                              helper=self.helper_train,
                                                              output_layer=self.fc_layer
                                                              )
            self.decoder_train_logits, _, _ = s2s.dynamic_decode(decoder=self.decoder_train
                                                                 )

            # for infer
            self.start_tokens = tf.fill([self.batch_size], 0)
            self.end_tokens = 0
            self.helper_infer = contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings_trainable,
                                                                      start_tokens=self.start_tokens,
                                                                      end_token=self.end_tokens)
            self.decoder_infer = contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                              initial_state=self.encoder_final_state,
                                                              helper=self.helper_infer,
                                                              output_layer=self.fc_layer)
            self.decoder_infer_logits, _, _ = s2s.dynamic_decode(self.decoder_infer,
                                                                 maximum_iterations=20
                                                                 )

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.targets = self.decoder_targets
            # self.targets = tf.one_hot(self.targets, depth=self.vocab_size, dtype=tf.float32)
            self.logits_flat = self.decoder_train_logits.rnn_output
            # tf.losses.softmax_cross_entropy
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.targets,
                                                               logits=self.logits_flat)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope("summaries_seq2seq"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def _create_log(self):
        LOG_FILE = './log/seq2seq.log'
        handler = logging.FileHandler(LOG_FILE, mode='w')  # 实例化handler
        fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
        formatter = logging.Formatter(fmt)  # 实例化formatter
        handler.setFormatter(formatter)  # 为handler添加formatter
        self.logger = logging.getLogger('seq2seqlogger')  # 获取名为tst的logger
        self.logger.addHandler(handler)  # 为logger添加handler
        self.logger.setLevel(logging.DEBUG)

    def _build_graph(self):
        self._create_placeholder()
        self._create_embedding()
        self._create_seq2seq()
        self._create_loss()
        if self.is_train:
            self._create_summaries()
            self._create_log()
        print("seq2seq graph built")

    def _run(self, epoch, num_train_steps, batches, skip_steps, one_hot=None):
        if self.is_train:
            ckpt = tf.train.get_checkpoint_state(self.MODEL_FILE)
            with tf.Session() as sess:
                if ckpt and ckpt.model_checkpoint_path:
                    print(ckpt.model_checkpoint_path)
                    self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                    self.loss = tf.get_collection('loss')[0]
                    self.optimizer = tf.get_collection('optimizer')[0]
                    self.summary_op = tf.get_collection('summary')[0]
                    print("the model has been successfully restored")

                else:
                    self._build_graph()
                    self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
                    sess.run(tf.global_variables_initializer())
                    tf.add_to_collection('loss', self.loss)
                    tf.add_to_collection('optimizer', self.optimizer)
                    tf.add_to_collection('summary', self.summary_op)
                    print("the model has been built")
                print("start training seq2seq model")
                writer = tf.summary.FileWriter('./graphs/seq2seq', sess.graph)
                for i in range(epoch):
                    self.total_loss = 0.0
                    for index in range(num_train_steps):
                        print("epoch: %d at train step: %d" % (i, index + 1))
                        self.global_step += self.batch_size
                        encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length = next(batches)
                        decoder_length = [30 for i in range(self.batch_size)]
                        feed_dict = {
                            self.decoder_targets: decoder_targets,
                            self.decoder_length: decoder_length,
                            self.encoder_inputs: encoder_inputs,
                            self.decoder_inputs: decoder_inputs
                        }

                        loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op],
                                                          feed_dict=feed_dict)
                        self.total_loss += loss_batch
                        writer.add_summary(summary, global_step=self.global_step.eval())
                        if (index + 1) % skip_steps == 0:
                            self.logger.debug('seq2seq average loss at epoch {} step {} : {:3.9f}'.format(i, index + 1,
                                                                                                          self.total_loss / skip_steps))
                            self.total_loss = 0.0
                    self.saver.save(sess, self.MODEL_FILE + 'model.ckpt', global_step=self.global_step)
                    self.logger.debug("seq2seq trained,model saved at epoch {}".format(i))
        else:
            self._build_graph()
            self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
            ckpt = tf.train.get_checkpoint_state(self.MODEL_FILE)
            with tf.Session() as sess:
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                    print("the model has been successfully restored")
                    for index in range(num_train_steps):
                        encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length = next(batches)
                        decoder_length = [30 for i in range(self.batch_size)]
                        feed_dict = {
                            self.decoder_targets: decoder_targets,
                            self.decoder_length: decoder_length,
                            self.encoder_inputs: encoder_inputs,
                            self.decoder_inputs: decoder_inputs
                        }

                        file = open("./infer/output.txt", "w")

                        for test_index in range(self.batch_size):
                            file.write("- group %d\n" % (test_index + 1))

                            file.write("     - infer headline: \n")
                            prediction = sess.run(self.decoder_infer_logits, feed_dict=feed_dict)
                            prediction = prediction.sample_id
                            # prediction = prediction[2]
                            answer = [one_hot[i] for i in prediction[test_index]]
                            output = "        "
                            for i in answer:
                                if i != "UNK":
                                    output += i
                                    output += " "
                            file.write(output)
                            file.write("\n")

                            # print("logits_flat")
                            # logits_flat = sess.run(self.decoder_train_logits, feed_dict=feed_dict)
                            # logits_flat = logits_flat.rnn_output
                            # logits_flat = np.argmax(logits_flat, 2)
                            # # logits_flat = logits_flat[2]
                            # answer = [one_hot[i] for i in logits_flat[test_index]]
                            # print(answer)

                            file.write("     - targets: \n")
                            targets = sess.run(self.decoder_targets, feed_dict=feed_dict)
                            # targets = targets[2]
                            answer = [one_hot[i] for i in targets[test_index]]
                            output = "        "
                            for i in answer:
                                if i != "UNK":
                                    output += i
                                    output += " "
                            file.write(output)
                            file.write("\n")
                            print("output %d finished" % test_index)

                        file.close()
                        print("infer file updated")
                else:
                    print("model restored failed")
                    pass
