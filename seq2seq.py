import tensorflow as tf
import logging.config
import tensorflow.contrib.seq2seq as s2s
import tensorflow.contrib as contrib
import tensorflow.contrib.rnn as rnn


class Seq2seqModel:
    def __init__(self, vocab_size, embed_size, encoder_hidden_units, decoder_hidden_units, batch_size,
                 embed_matrix_init, encoder_layers, learning_rate_initial, keep_prob, core):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        self.batch_size = batch_size
        self.embed_matrix_init = embed_matrix_init
        self.encoder_layers = encoder_layers
        self.learning_rate_initial = learning_rate_initial
        self.keep_prob = keep_prob
        self.core = core
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.global_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_epoch')
        self.MODEL_FILE = './model/'

    def _create_placeholder(self):
        with tf.name_scope("data_seq2seq"):
            self.encoder_inputs = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='encoder_inputs')
            self.decoder_inputs = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='decoder_inputs')
            self.decoder_targets = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='decoder_targets')
            self.decoder_length = tf.placeholder(shape=self.batch_size, dtype=tf.int32, name='decoder_length')
            self.encoder_length = tf.placeholder(shape=self.batch_size, dtype=tf.int32, name='encoder_length')

    def _create_embedding(self):
        # self.embeddings_trainable = tf.Variable(initial_value=self.embed_matrix_init, name='word_embedding_train')
        # fix embedding_layer, do not train with the model
        self.embeddings_untrainable = tf.Variable(initial_value=self.embed_matrix_init, trainable=False,
                                                  name='word_embedding_fixed')
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings_untrainable, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings_untrainable, self.decoder_inputs)

    def _create_blstmcell(self, layer_i):
        with tf.variable_scope('lstm_layer%i' % layer_i, reuse=tf.AUTO_REUSE):
            # if layer_i == 1:
            #     self.encoder_hidden_units *= 2
            cell_fw = rnn.LSTMCell(
                num_units=self.encoder_hidden_units,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=114),
                state_is_tuple=True)
            cell_bw = rnn.LSTMCell(
                num_units=self.encoder_hidden_units,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=133),
                state_is_tuple=True)

            cell_fw = rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=self.keep_prob)
            cell_bw = rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=self.keep_prob)
        return cell_fw, cell_bw

    def _create_bgrucell(self):
        with tf.variable_scope("bgru_layer"):
            cell_fw = tf.nn.rnn_cell.GRUCell(
                num_units=self.encoder_hidden_units,
                kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=133),
                bias_initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=18))
            cell_bw = tf.nn.rnn_cell.GRUCell(
                num_units=self.encoder_hidden_units,
                kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=114),
                bias_initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=107))
        return cell_fw, cell_bw

    def _create_seq2seq(self):

        if self.core == "blstm":
            # Mutilayer  BLSTM Encoder
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                for layer_i in range(self.encoder_layers):
                    cell_fw, cell_bw = self._create_blstmcell(layer_i)
                    (self.encoder_inputs_embedded, self.encoder_final_state) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cell_fw,
                        cell_bw=cell_bw,
                        inputs=self.encoder_inputs_embedded,
                        dtype=tf.float32)
                    self.encoder_inputs_embedded = tf.add_n(self.encoder_inputs_embedded)
                    # if self.is_train == 0:
                    #     self.encoder_inputs_embedded = tf.multiply(self.encoder_inputs_embedded, self.keep_prob)

                self.encoder_final_state_c = tf.concat(
                    (self.encoder_final_state[0].c, self.encoder_final_state[1].c), 1)
                self.encoder_final_state_h = tf.concat(
                    (self.encoder_final_state[0].h, self.encoder_final_state[1].h), 1)
                self.encoder_final_state = contrib.rnn.LSTMStateTuple(
                    c=self.encoder_final_state_c,
                    h=self.encoder_final_state_h)

            # Basic Attention based LSTM Decoder(train and infer)
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                self.decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.decoder_hidden_units,
                                                            state_is_tuple=True)

                self.attention_state = self.encoder_inputs_embedded
                self.attention_mechanism = contrib.seq2seq.LuongAttention(num_units=self.decoder_hidden_units,
                                                                          memory=self.attention_state,
                                                                          memory_sequence_length=self.encoder_length)
                self.attn_cell = contrib.seq2seq.AttentionWrapper(cell=self.decoder_cell,
                                                                  attention_mechanism=self.attention_mechanism,
                                                                  name="decoder_attention_cell",
                                                                  alignment_history=False
                                                                  )
                self.fc_layer = tf.layers.Dense(self.vocab_size, name='dense_layer')

                # for train
                with tf.variable_scope('decoder_train', reuse=tf.AUTO_REUSE):
                    self.helper_train = contrib.seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                                       sequence_length=self.decoder_length)
                    self.decoder_initial_state = self.attn_cell.zero_state(self.batch_size, dtype=tf.float32).clone(
                        cell_state=self.encoder_final_state)
                    self.decoder_train = contrib.seq2seq.BasicDecoder(cell=self.attn_cell,
                                                                      initial_state=self.decoder_initial_state,
                                                                      helper=self.helper_train,
                                                                      output_layer=self.fc_layer
                                                                      )
                    self.decoder_train_logits, _, _ = s2s.dynamic_decode(decoder=self.decoder_train
                                                                         )

                # for infer
                with tf.variable_scope('decoder_infer', reuse=tf.AUTO_REUSE):
                    self.start_tokens = tf.tile([19654], [self.batch_size])
                    self.end_tokens = 19655
                    self.helper_infer = contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings_trainable,
                                                                              start_tokens=self.start_tokens,
                                                                              end_token=self.end_tokens)
                    self.decoder_infer = contrib.seq2seq.BasicDecoder(cell=self.attn_cell,
                                                                      initial_state=self.decoder_initial_state,
                                                                      helper=self.helper_infer,
                                                                      output_layer=self.fc_layer)
                    self.decoder_infer_logits, _, _ = s2s.dynamic_decode(decoder=self.decoder_infer,
                                                                         maximum_iterations=20
                                                                         )

        elif self.core == "bgru":
            # single layer bgru encoder
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                inputs = self.encoder_inputs_embedded
                cell_fw, cell_bw = self._create_bgrucell()
                with tf.variable_scope(None, default_name="encoder"):
                    (output, self.encoder_final_state) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cell_fw,
                        cell_bw=cell_bw,
                        inputs=inputs,
                        dtype=tf.float32)

                self.encoder_final_state = tf.concat(self.encoder_final_state, 1)

            # basic gru Decoder for train and infer
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                self.decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=self.decoder_hidden_units,
                                                           name='decoder_cell')
                self.attention_state = self.encoder_inputs_embedded
                self.attention_mechanism = contrib.seq2seq.LuongAttention(num_units=self.decoder_hidden_units,
                                                                          memory=self.attention_state,
                                                                          memory_sequence_length=self.encoder_length)
                self.attn_cell = contrib.seq2seq.AttentionWrapper(cell=self.decoder_cell,
                                                                  attention_mechanism=self.attention_mechanism,
                                                                  name="decoder_attention_cell",
                                                                  alignment_history=False
                                                                  )
                self.fc_layer = tf.layers.Dense(self.vocab_size, name='dense_layer')

                with tf.variable_scope('decoder_train', reuse=tf.AUTO_REUSE):
                    # for train
                    self.helper_train = contrib.seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                                       sequence_length=self.decoder_length)
                    self.decoder_initial_state = self.attn_cell.zero_state(self.batch_size, dtype=tf.float32).clone(
                        cell_state=self.encoder_final_state)
                    self.decoder_train = contrib.seq2seq.BasicDecoder(cell=self.attn_cell,
                                                                      initial_state=self.decoder_initial_state,
                                                                      helper=self.helper_train,
                                                                      output_layer=self.fc_layer
                                                                      )
                    self.decoder_train_logits, _, _ = s2s.dynamic_decode(decoder=self.decoder_train
                                                                         )
                with tf.variable_scope('decoder_infer', reuse=tf.AUTO_REUSE):
                    # for infer
                    self.start_tokens = tf.fill([self.batch_size], 2000)
                    self.end_tokens = 2001
                    self.helper_infer = contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings_untrainable,
                                                                              start_tokens=self.start_tokens,
                                                                              end_token=self.end_tokens)
                    self.decoder_infer = contrib.seq2seq.BasicDecoder(cell=self.attn_cell,
                                                                      initial_state=self.decoder_initial_state,
                                                                      helper=self.helper_infer,
                                                                      output_layer=self.fc_layer)
                    self.decoder_infer_logits, _, _ = s2s.dynamic_decode(self.decoder_infer,
                                                                         maximum_iterations=20
                                                                         )

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.targets = self.decoder_targets
            self.logits_train = self.decoder_train_logits.rnn_output
            self.logits_infer = self.decoder_infer_logits.rnn_output
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.targets,
                                                               logits=self.logits_train)
            self.loss_infer = tf.losses.sparse_softmax_cross_entropy(labels=self.targets,
                                                                     logits=self.logits_infer)
            self.learning_rate = tf.train.exponential_decay(self.learning_rate_initial,
                                                            global_step=self.global_epoch,
                                                            decay_steps=100, decay_rate=0.995)
            self.add_global = self.global_epoch.assign_add(1)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_initial).minimize(self.loss)
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope("summaries_seq2seq"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def _create_log(self):
        log_file = './log/seq2seq.log'
        handler = logging.FileHandler(log_file, mode='w')  # 实例化handler
        fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
        formatter = logging.Formatter(fmt)  # 实例化formatter
        handler.setFormatter(formatter)  # 为handler添加formatter
        self.logger = logging.getLogger('seq2seqlogger')  # 获取名为tst的logger
        self.logger.addHandler(handler)  # 为logger添加handler
        self.logger.setLevel(logging.DEBUG)

    def build_graph(self):
        self._create_placeholder()
        self._create_embedding()
        self._create_seq2seq()
        self._create_loss()
        self._create_summaries()
        self._create_log()

    def first_train(self, epoch_total, num_train_steps, batches, skip_steps):

        # limit the usage of gpu
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
            saver = tf.train.Saver(max_to_keep=5)
            sess.run(tf.global_variables_initializer())
            print("start training seq2seq model in [%s] mode" % self.core)
            writer = tf.summary.FileWriter('./graphs/seq2seq', sess.graph)
            for i in range(epoch_total):
                total_loss = 0.0
                epoch_index, lr = sess.run([self.add_global, self.learning_rate])
                self.logger.debug("at epoch {} the learning rate is {}".format(epoch_index, lr))
                self.logger.debug("--------------------------------------------------------")

                # save last batch in each epoch for validate
                for index in range(num_train_steps - 1):
                    print("batch: %d at epoch: %d" % (index + 1, epoch_index))
                    self.global_step += self.batch_size
                    encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length = next(batches)
                    decoder_length_batch = [decoder_length for i in range(self.batch_size)]
                    encoder_length_batch = [encoder_length for i in range(self.batch_size)]

                    feed_dict = {
                        self.decoder_targets: decoder_targets,
                        self.decoder_length: decoder_length_batch,
                        self.encoder_inputs: encoder_inputs,
                        self.decoder_inputs: decoder_inputs,
                        self.encoder_length: encoder_length_batch
                    }

                    loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op],
                                                      feed_dict=feed_dict)
                    total_loss += loss_batch
                    writer.add_summary(summary, global_step=self.global_step.eval())
                    if (index + 1) % skip_steps == 0:
                        self.logger.debug('loss at epoch {} batch {} : {:3.9f}'.format(epoch_index, index + 1,
                                                                                       total_loss / skip_steps))
                        total_loss = 0.0

                # use last batch to assess generalization
                print("epoch: %d validation" % epoch_index)
                self.global_step += self.batch_size
                encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length = next(batches)
                decoder_length_batch = [decoder_length for i in range(self.batch_size)]
                encoder_length_batch = [encoder_length for i in range(self.batch_size)]

                feed_dict = {
                    self.decoder_targets: decoder_targets,
                    self.decoder_length: decoder_length_batch,
                    self.encoder_inputs: encoder_inputs,
                    self.decoder_inputs: decoder_inputs,
                    self.encoder_length: encoder_length_batch
                }

                loss_batch_validate, _ = sess.run([self.loss, self.optimizer],
                                                  feed_dict=feed_dict)
                self.logger.debug("validate loss at epoch {} :{:3.9f}".format(epoch_index, loss_batch_validate))

                saver.save(sess=sess,
                           save_path=self.MODEL_FILE + 'model.ckpt',
                           global_step=self.global_step,
                           write_meta_graph=True)
                self.logger.debug("seq2seq trained,model saved at epoch {}\n".format(epoch_index))

    def continue_train(self, epoch_total, num_train_steps, batches, skip_steps):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        ckpt = tf.train.get_checkpoint_state(self.MODEL_FILE)
        if ckpt and ckpt.model_checkpoint_path:
            print("found model,continue training")
            with tf.Session(config=config) as sess:
                # saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
                saver = tf.train.Saver()
                # saver.restore(sess, tf.train.latest_checkpoint(self.MODEL_FILE))
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("start training seq2seq model in [%s] mode" % self.core)
                writer = tf.summary.FileWriter('./graphs/seq2seq', sess.graph)
                for i in range(epoch_total):
                    total_loss = 0.0
                    epoch_index, lr = sess.run([self.add_global, self.learning_rate])
                    self.logger.debug("at epoch {} the learning rate is {}".format(epoch_index, lr))
                    self.logger.debug("--------------------------------------------------------")

                    # save last batch in each epoch for validate
                    for index in range(num_train_steps - 1):
                        print("batch: %d at epoch: %d" % (index + 1, epoch_index))
                        self.global_step += self.batch_size
                        encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length = next(batches)
                        decoder_length_batch = [decoder_length for i in range(self.batch_size)]
                        encoder_length_batch = [encoder_length for i in range(self.batch_size)]

                        feed_dict = {
                            self.decoder_targets: decoder_targets,
                            self.decoder_length: decoder_length_batch,
                            self.encoder_inputs: encoder_inputs,
                            self.decoder_inputs: decoder_inputs,
                            self.encoder_length: encoder_length_batch
                        }

                        loss_batch, _, summary = sess.run([self.loss, self.optimizer, self.summary_op],
                                                          feed_dict=feed_dict)
                        total_loss += loss_batch
                        writer.add_summary(summary, global_step=self.global_step.eval())
                        if (index + 1) % skip_steps == 0:
                            self.logger.debug('loss at epoch {} batch {} : {:3.9f}'.format(epoch_index, index + 1,
                                                                                           total_loss / skip_steps))
                            total_loss = 0.0

                    # use last batch to assess generalization
                    print("epoch: %d validation" % epoch_index)
                    self.global_step += self.batch_size
                    encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length = next(batches)
                    decoder_length_batch = [decoder_length for i in range(self.batch_size)]
                    encoder_length_batch = [encoder_length for i in range(self.batch_size)]

                    feed_dict = {
                        self.decoder_targets: decoder_targets,
                        self.decoder_length: decoder_length_batch,
                        self.encoder_inputs: encoder_inputs,
                        self.decoder_inputs: decoder_inputs,
                        self.encoder_length: encoder_length_batch
                    }

                    loss_batch_validate, _ = sess.run([self.loss, self.optimizer],
                                                      feed_dict=feed_dict)
                    self.logger.debug("validate loss at epoch {} :{:3.9f}".format(epoch_index, loss_batch_validate))

                    saver.save(sess=sess,
                               save_path=self.MODEL_FILE + 'model.ckpt',
                               global_step=self.global_step,
                               write_meta_graph=False)
                    self.logger.debug("seq2seq trained,model saved at epoch {}\n".format(epoch_index))
        else:
            print("model not found,check your saved model")
