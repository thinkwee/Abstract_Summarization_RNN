import tensorflow as tf
import logging.config
import tensorflow.contrib.seq2seq as s2s
import tensorflow.contrib as contrib
import tensorflow.contrib.rnn as rnn
import shuffle


class Seq2seqModel:
    def __init__(self, vocab_size, embed_size, encoder_hidden_units, decoder_hidden_units, batch_size,
                 embed_matrix_init, encoder_layers, learning_rate_initial, keep_prob, rnn_core, start_token_id,
                 end_token_id, num_layers, is_train, grad_clip, is_continue):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        self.batch_size = batch_size
        self.embed_matrix_init = embed_matrix_init
        self.encoder_layers = encoder_layers
        self.learning_rate_initial = learning_rate_initial
        self.keep_prob = keep_prob
        self.core = rnn_core
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.global_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_epoch')
        self.MODEL_FILE = './model/'
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.grad_clip = grad_clip
        self.is_continue = is_continue

    def _create_placeholder(self):
        with tf.name_scope("data_seq2seq"):
            self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
            self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
            self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
            self.decoder_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_length')
            self.encoder_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_length')
            self.decoder_max_iter = tf.placeholder(shape=(), dtype=tf.int32, name='encoder_length')

    def _create_embedding(self):
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size]))
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

    def _create_blstmcell(self, layer_i):
        with tf.variable_scope('lstm_layer%i' % layer_i, reuse=tf.AUTO_REUSE):
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
            cell_fw = contrib.cudnn_rnn.CudnnCompatibleGRUCell(
                num_units=self.encoder_hidden_units,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                   stddev=0.1))
            cell_bw = contrib.cudnn_rnn.CudnnCompatibleGRUCell(
                num_units=self.encoder_hidden_units,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                   stddev=0.1))
        return cell_fw, cell_bw

    def _create_blstm_seq2seq(self):
        # TODO:need to correct
        # BiLSTM Encoder
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            cell_fw, cell_bw = self._create_blstmcell()
            (self.encoder_inputs_embedded, self.encoder_final_state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.encoder_inputs_embedded,
                dtype=tf.float32)
            self.encoder_inputs_embedded = tf.add_n(self.encoder_inputs_embedded)

            self.encoder_final_state_c = tf.concat(
                (self.encoder_final_state[0].c, self.encoder_final_state[1].c), 1)
            self.encoder_final_state_h = tf.concat(
                (self.encoder_final_state[0].h, self.encoder_final_state[1].h), 1)
            self.encoder_final_state = contrib.rnn.LSTMStateTuple(
                c=self.encoder_final_state_c,
                h=self.encoder_final_state_h)

        # Basic LSTM Decoder(train and infer)
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            self.decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.decoder_hidden_units,
                                                        state_is_tuple=True)
            self.fc_layer = tf.layers.Dense(self.vocab_size, name='dense_layer')

            # for train
            with tf.variable_scope('decoder_train', reuse=tf.AUTO_REUSE):
                self.helper_train = contrib.seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                                   sequence_length=self.decoder_length)
                self.decoder_train = contrib.seq2seq.BasicDecoder(cell=self.attn_cell,
                                                                  initial_state=self.decoder_initial_state,
                                                                  helper=self.helper_train,
                                                                  output_layer=self.fc_layer
                                                                  )
                self.decoder_train_logits, _, _ = s2s.dynamic_decode(decoder=self.decoder_train
                                                                     )
            # for infer
            with tf.variable_scope('decoder_infer', reuse=tf.AUTO_REUSE):
                self.start_tokens = tf.tile([self.start_token_id], [self.batch_size])
                self.helper_infer = contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                                          start_tokens=self.start_tokens,
                                                                          end_token=self.end_token_id)
                self.decoder_infer = contrib.seq2seq.BasicDecoder(cell=self.attn_cell,
                                                                  initial_state=self.decoder_initial_state,
                                                                  helper=self.helper_infer,
                                                                  output_layer=self.fc_layer)
                self.decoder_infer_logits, _, _ = s2s.dynamic_decode(decoder=self.decoder_infer,
                                                                     maximum_iterations=self.decoder_max_iter
                                                                     )

    def _create_bgru_seq2seq(self):
        # single layer bgru encoder
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE, initializer=tf.initializers.orthogonal):
            inputs = self.encoder_inputs_embedded
            cell_fw, cell_bw = self._create_bgrucell()
            with tf.variable_scope(None, default_name="encoder"):
                (output, self.encoder_final_state) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=inputs,
                    dtype=tf.float32,
                    sequence_length=self.encoder_length,
                    parallel_iterations=32)

            self.encoder_final_state = tf.concat(self.encoder_final_state, 1)

        # basic gru Decoder for train and infer
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE, initializer=tf.initializers.orthogonal):
            self.decoder_cell = contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=self.decoder_hidden_units,
                                                                         kernel_initializer=tf.truncated_normal_initializer(
                                                                             mean=0.0,
                                                                             stddev=0.1))
            self.fc_layer = tf.layers.Dense(self.vocab_size,
                                            kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                               stddev=0.1),
                                            name='dense_layer')

            with tf.variable_scope('decoder_train', reuse=tf.AUTO_REUSE):
                # for train
                self.helper_train = contrib.seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                                   sequence_length=self.decoder_length)
                self.decoder_train = contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                                  initial_state=self.encoder_final_state,
                                                                  helper=self.helper_train,
                                                                  output_layer=self.fc_layer
                                                                  )
                self.decoder_train_output, _, _ = s2s.dynamic_decode(decoder=self.decoder_train,
                                                                     maximum_iterations=self.decoder_max_iter)

            with tf.variable_scope('decoder_infer', reuse=tf.AUTO_REUSE):
                # for infer
                self.start_tokens = tf.fill([self.batch_size], self.start_token_id)
                self.helper_infer = contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                                          start_tokens=self.start_tokens,
                                                                          end_token=self.end_token_id)
                self.decoder_infer = contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                                  initial_state=self.encoder_final_state,
                                                                  helper=self.helper_infer,
                                                                  output_layer=self.fc_layer)
                self.decoder_infer_output, _, _ = s2s.dynamic_decode(self.decoder_infer,
                                                                     impute_finished=True,
                                                                     maximum_iterations=self.decoder_max_iter
                                                                     )

    def _create_gru_seq2seq(self):
        # single layer bgru encoder
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            inputs = self.encoder_inputs_embedded
            cell = tf.nn.rnn_cell.GRUCell(num_units=self.encoder_hidden_units)
            with tf.variable_scope(None, default_name="encoder"):
                (output, self.encoder_final_state) = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=inputs,
                    dtype=tf.float32,
                    sequence_length=self.encoder_length,
                    parallel_iterations=32)

        # basic gru Decoder for train and infer
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            self.decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=self.decoder_hidden_units)
            self.fc_layer = tf.layers.Dense(self.vocab_size,
                                            kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                               stddev=0.1),
                                            name='dense_layer')

            with tf.variable_scope('decoder_train', reuse=tf.AUTO_REUSE):
                # for train
                self.helper_train = contrib.seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                                   sequence_length=self.decoder_length)
                self.decoder_train = contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                                  initial_state=self.encoder_final_state,
                                                                  helper=self.helper_train,
                                                                  output_layer=self.fc_layer
                                                                  )
                self.decoder_train_output, _, _ = s2s.dynamic_decode(decoder=self.decoder_train,
                                                                     maximum_iterations=self.decoder_max_iter)

            with tf.variable_scope('decoder_infer', reuse=tf.AUTO_REUSE):
                # for infer
                self.start_tokens = tf.fill([self.batch_size], self.start_token_id)
                self.helper_infer = contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                                          start_tokens=self.start_tokens,
                                                                          end_token=self.end_token_id)
                self.decoder_infer = contrib.seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                                  initial_state=self.encoder_final_state,
                                                                  helper=self.helper_infer,
                                                                  output_layer=self.fc_layer)
                self.decoder_infer_output, _, _ = s2s.dynamic_decode(self.decoder_infer,
                                                                     impute_finished=True,
                                                                     maximum_iterations=self.decoder_max_iter
                                                                     )

    def _create_attention_seq2seq(self):
        # TODO:need to correct
        # single layer bgru encoder
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            inputs = self.encoder_inputs_embedded
            cell_fw, cell_bw = self._create_bgrucell()
            with tf.variable_scope(None, default_name="encoder"):
                (output, self.encoder_final_state) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=inputs,
                    dtype=tf.float32,
                    sequence_length=self.encoder_length,
                    parallel_iterations=128)

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
                self.start_tokens = tf.fill([self.batch_size], self.start_token_id)
                self.helper_infer = contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                                          start_tokens=self.start_tokens,
                                                                          end_token=self.end_token_id)
                self.decoder_infer = contrib.seq2seq.BasicDecoder(cell=self.attn_cell,
                                                                  initial_state=self.decoder_initial_state,
                                                                  helper=self.helper_infer,
                                                                  output_layer=self.fc_layer)
                self.decoder_infer_logits, _, _ = s2s.dynamic_decode(self.decoder_infer,
                                                                     maximum_iterations=self.decoder_max_iter
                                                                     )

    def _create_seq2seq(self):

        if self.core == "blstm":
            self._create_blstm_seq2seq()

        elif self.core == "bgru":
            self._create_bgru_seq2seq()

        elif self.core == "bgru_attetion":
            self._create_attention_seq2seq()

        elif self.core == "gru":
            self._create_gru_seq2seq()

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.targets = tf.identity(self.decoder_targets)
            self.logits_train = tf.identity(self.decoder_train_output.rnn_output, 'training_logits')
            self.logits_infer = tf.identity(self.decoder_infer_output.rnn_output, 'infer_logits')

            # use mask to achieve dynamic loss calculate,but first you should make targets be padded
            masks_train = tf.sequence_mask(self.decoder_length, self.decoder_max_iter, dtype=tf.float32, name='masks')
            self.loss = s2s.sequence_loss(targets=self.targets,
                                          logits=self.logits_train,
                                          weights=masks_train)

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            # gradient clip
            train_variable = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_variable), self.grad_clip)

            self.learning_rate = tf.train.exponential_decay(self.learning_rate_initial,
                                                            global_step=self.global_epoch,
                                                            decay_steps=1000, decay_rate=0.995)

            self.add_global_epoch = self.global_epoch.assign_add(1)
            self.add_global_step = self.global_step.assign_add(self.batch_size)

            # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_initial, momentum=0.9)

            self.train_op = self.optimizer.apply_gradients(zip(grads, train_variable))

    def _create_summaries(self):
        with tf.name_scope("summaries_seq2seq"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def _create_log(self):
        log_file = './log/seq2seq.log'
        handler = logging.FileHandler(log_file, mode='w')
        fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        self.logger = logging.getLogger('seq2seqlogger')
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def build_graph(self):
        self._create_placeholder()
        self._create_embedding()
        self._create_seq2seq()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        self._create_log()

    def train(self, epoch_total, num_train_steps, batches, skip_steps):

        # limit the usage of gpu
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7

        config.gpu_options.allow_growth = True

        if self.is_continue:
            ckpt = tf.train.get_checkpoint_state(self.MODEL_FILE)
            if ckpt and ckpt.model_checkpoint_path:
                print("found model,continue training")
            else:
                print("model not found,check your saved model")

        # lock the graph for the sake of lazy loading
        graph = tf.get_default_graph
        tf.Graph.finalize(graph)
        min_validate_loss = 32768.0

        with tf.Session(config=config) as sess:

            if self.is_continue:
                saver = tf.train.Saver()
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("continue training seq2seq model in [%s] mode" % self.core)
            else:
                saver = tf.train.Saver(max_to_keep=1)
                sess.run(tf.global_variables_initializer())
                print("start training seq2seq model in [%s] mode" % self.core)

            writer = tf.summary.FileWriter('./graphs/seq2seq', sess.graph)

            for i in range(epoch_total):
                total_loss = 0.0
                epoch_index, lr = sess.run([self.add_global_epoch, self.learning_rate])
                # self.logger.debug("at epoch {} the learning rate is {}".format(epoch_index, lr))
                self.logger.debug("--------------------------------------------------------")

                # save last batch in each epoch for validation
                for index in range(num_train_steps):
                    self.global_step = sess.run(self.add_global_step)
                    encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length, decoder_max_iter = next(
                        batches)
                    feed_dict = {
                        self.decoder_targets: decoder_targets,
                        self.decoder_length: decoder_length,
                        self.encoder_inputs: encoder_inputs,
                        self.decoder_inputs: decoder_inputs,
                        self.encoder_length: encoder_length,
                        self.decoder_max_iter: decoder_max_iter
                    }
                    if index == num_train_steps - 1:
                        loss_batch_validate, = sess.run([self.loss],
                                                        feed_dict=feed_dict)
                        self.logger.debug("validate loss at epoch {} :{:3.9f}".format(epoch_index, loss_batch_validate))
                        print("epoch: %d validation: %9.9f" % (epoch_index, loss_batch_validate))

                        # save 5 minimum validate loss model
                        # if min_validate_loss > loss_batch_validate:
                        #     min_validate_loss = loss_batch_validate
                        saver.save(sess=sess,
                                   save_path=self.MODEL_FILE + 'model.ckpt',
                                   global_step=self.global_step,
                                   write_meta_graph=True)
                        self.logger.debug(
                            "seq2seq trained,model saved at epoch {},validate loss is {}\n".format(epoch_index,
                                                                                                   min_validate_loss))
                    else:
                        loss_batch, _, summary = sess.run([self.loss, self.train_op, self.summary_op],
                                                          feed_dict=feed_dict)
                        total_loss += loss_batch
                        writer.add_summary(summary, global_step=self.global_step)
                        if (index + 1) % skip_steps == 0:
                            self.logger.debug('loss at epoch {} batch {} : {:3.9f}'.format(epoch_index, index + 1,
                                                                                           total_loss / skip_steps))
                            print('loss at epoch %d batch %d : %9.9f' % (epoch_index, index + 1,
                                                                         total_loss / skip_steps))
                            total_loss = 0.0
                shuffle.shuffle_train_data()

    def test(self, epoch, num_train_steps, batches, one_hot):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.MODEL_FILE)
        with tf.Session() as sess:
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("the model has been successfully restored")
                for _ in range(epoch):
                    for _ in range(num_train_steps):
                        encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length, decoder_max_iter = next(
                            batches)

                        feed_dict = {
                            self.decoder_targets: decoder_targets,
                            self.decoder_length: decoder_length,
                            self.encoder_inputs: encoder_inputs,
                            self.decoder_inputs: decoder_inputs,
                            self.encoder_length: encoder_length,
                            self.decoder_max_iter: decoder_max_iter
                        }

                        infer_output = sess.run(self.decoder_infer_output, feed_dict=feed_dict)
                        prediction_infer = infer_output.sample_id

                        train_output = sess.run(self.decoder_train_output, feed_dict=feed_dict)
                        prediction_train = train_output.sample_id

                        targets = sess.run(self.decoder_targets, feed_dict=feed_dict)

                        file = open("./infer/output.txt", "w")
                        for index in range(self.batch_size):

                            file.write("- group %d\n" % (index + 1))

                            file.write("     - infer headline: \n")
                            prediction_infer_single = prediction_infer[index]
                            answer = [one_hot[i] for i in prediction_infer_single]
                            output = "        "
                            for i in answer:
                                if i != 'UNK' and i != '_PAD':
                                    output += i
                                    output += " "
                            file.write(output)
                            file.write("\n")
                            file_create = open("./ROUGE/models/test" + str(index) + ".txt", "w")
                            file_create.writelines(output)
                            file_create.close()

                            file.write("     - train headline: \n")
                            prediction_train_single = prediction_train[index]
                            answer = [one_hot[i] for i in prediction_train_single]
                            output = "        "
                            for i in answer:
                                if i != 'UNK' and i != '_PAD':
                                    output += i
                                    output += " "
                            file.write(output)
                            file.write("\n")

                            file.write("     - targets: \n")
                            targets_single = targets[index]
                            answer = [one_hot[i] for i in targets_single]
                            output = "        "
                            for i in answer:
                                if i != 'UNK' and i != '_PAD':
                                    output += i
                                    output += " "
                            file.write(output)
                            file.write("\n")
                            print("output %d finished" % index)

                        file.close()
                        print("infer file updated")
            else:
                print("model restored failed")
                pass
