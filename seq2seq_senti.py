import tensorflow as tf
import logging.config
import tensorflow.contrib.seq2seq as s2s
import tensorflow.contrib as contrib
import shuffle
import sentiwordnet
import nltk
import numpy as np


# net_path = "./data/SentiWordNet.txt"
# np_dict = sentiwordnet.SentiWordNet(net_path)
# np_dict.infoextract()


class Seq2seqModel:
    def __init__(self, vocab_size, embed_size, encoder_hidden_units, decoder_hidden_units, batch_size,
                 embed_matrix_init, learning_rate_initial, keep_prob, rnn_core, start_token_id,
                 end_token_id, num_layers, grad_clip, is_continue, one_hot):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        self.batch_size = batch_size
        self.embed_matrix_init = embed_matrix_init
        self.learning_rate_initial = learning_rate_initial
        self.keep_prob = keep_prob
        self.core = rnn_core
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.global_epoch = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='global_epoch')
        self.MODEL_FILE = './model/'
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.grad_clip = grad_clip
        self.is_continue = is_continue
        self.num_layers = num_layers
        self.one_hot = one_hot

    def _create_placeholder(self):
        with tf.name_scope("data_seq2seq"):
            self.encoder_inputs = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='encoder_inputs')
            self.decoder_inputs = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='decoder_inputs')
            self.decoder_targets = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32, name='decoder_targets')
            self.decoder_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_length')
            self.encoder_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_length')
            self.decoder_max_iter = tf.placeholder(shape=(), dtype=tf.int32, name='encoder_length')
            self.article_sen_vec = tf.placeholder(shape=(self.batch_size, None), dtype=tf.float32,
                                                  name="article_sentiment_vector")

    def _create_embedding(self):
        self.embeddings_encoder = tf.Variable(initial_value=self.embed_matrix_init, trainable=True)
        self.embeddings_decoder = tf.Variable(initial_value=self.embed_matrix_init, trainable=True)
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings_encoder, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings_decoder, self.decoder_inputs)

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

    def _create_bgru_seq2seq(self):
        # single layer bgru encoder
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            inputs = self.encoder_inputs_embedded
            cells_fw = []
            cells_bw = []
            for _ in range(self.num_layers):
                cell_fw, cell_bw = self._create_bgrucell()
                cell_fw = contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=self.keep_prob)
                cell_bw = contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=self.keep_prob)
                cells_fw.append(cell_fw)
                cells_bw.append(cell_bw)
            _, encoder_final_state_fw, encoder_final_state_bw = contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=cells_fw,
                cells_bw=cells_bw,
                inputs=inputs,
                dtype=tf.float32,
                sequence_length=self.encoder_length,
                parallel_iterations=32)
            self.encoder_final_state = tf.concat(axis=1, values=[encoder_final_state_fw[self.num_layers - 1],
                                                                 encoder_final_state_bw[self.num_layers - 1]])

            self.encoder_final_state = tf.concat(axis=1, values=[self.encoder_final_state, self.article_sen_vec])

        # basic gru Decoder for train and infer
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
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
                self.helper_infer = contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings_decoder,
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

    def _create_seq2seq(self):

        if self.core == "bgru":
            self._create_bgru_seq2seq()
        else:
            print("only senti_bgru is provided")

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.targets = tf.identity(self.decoder_targets)
            self.logits_train = tf.identity(self.decoder_train_output.rnn_output, 'training_logits')

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

            # exponential_decay learning rate
            # self.learning_rate = tf.train.exponential_decay(self.learning_rate_initial,
            #                                                 global_step=self.global_epoch,
            #                                                 decay_steps=1000, decay_rate=0.995)

            # sin learning rate
            sin_value = tf.sin(tf.multiply(3.14 / 5.0, self.global_epoch))
            self.learning_rate = tf.add(tf.multiply(0.1, sin_value), 0.11)

            self.add_global_epoch = self.global_epoch.assign_add(1.0)
            self.add_global_step = self.global_step.assign_add(self.batch_size)

            # SGD Optimizer
            # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            # self.train_op = self.optimizer.minimize(self.loss)

            # Momentum Optimizer
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
            # self.train_op = self.optimizer.apply_gradients(zip(grads, train_variable))

            # Adam Optimizer
            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.loss)

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
                saver = tf.train.Saver(max_to_keep=3)
                sess.run(tf.global_variables_initializer())
                print("start training seq2seq model in [%s] mode" % self.core)

            writer = tf.summary.FileWriter('./graphs/seq2seq', sess.graph)

            for i in range(epoch_total):
                shuffle.shuffle_senti_data()
                total_loss = 0.0
                epoch_index, lr = sess.run([self.add_global_epoch, self.learning_rate])
                self.logger.debug("at epoch {} the learning rate is {}".format(epoch_index, lr))
                print("learning rate is: %9.9f" % lr)
                self.logger.debug("--------------------------------------------------------")

                # save last batch in each epoch for validation
                for index in range(num_train_steps):
                    self.global_step = sess.run(self.add_global_step)
                    encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length, decoder_max_iter, article_sen_vec = next(
                        batches)
                    feed_dict = {
                        self.decoder_targets: decoder_targets,
                        self.decoder_length: decoder_length,
                        self.encoder_inputs: encoder_inputs,
                        self.decoder_inputs: decoder_inputs,
                        self.encoder_length: encoder_length,
                        self.decoder_max_iter: decoder_max_iter,
                        self.article_sen_vec: article_sen_vec
                    }
                    if index == num_train_steps - 1:
                        loss_batch_validate, = sess.run([self.loss],
                                                        feed_dict=feed_dict)
                        self.logger.debug("validate loss at epoch {} :{:3.9f}".format(epoch_index, loss_batch_validate))
                        print("epoch: %d validation: %9.9f\n" % (epoch_index, loss_batch_validate))

                        # save 5 minimum validate loss model
                        # if min_validate_loss > loss_batch_validate:
                        #     min_validate_loss = loss_batch_validate
                        if epoch_index % 2 == 0:
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

    def test(self, epoch, num_train_steps, batches):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.MODEL_FILE)
        with tf.Session() as sess:
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("the model has been successfully restored")
                file_senti_test = open("./infer/senti_test.txt", "w")

                for _ in range(epoch):
                    for index_num in range(num_train_steps):
                        encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length, decoder_max_iter, article_sen_vec = next(
                            batches)

                        feed_dict = {
                            self.decoder_targets: decoder_targets,
                            self.decoder_length: decoder_length,
                            self.encoder_inputs: encoder_inputs,
                            self.decoder_inputs: decoder_inputs,
                            self.encoder_length: encoder_length,
                            self.decoder_max_iter: decoder_max_iter,
                            self.article_sen_vec: article_sen_vec
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
                            answer = [self.one_hot[i] for i in prediction_infer_single]
                            output = "        "
                            for i in answer:
                                if i != 'UNK' and i != '_PAD':
                                    output += i
                                    output += " "
                            file.write(output)
                            file.write("\n")
                            file_senti_test.write(output)
                            file_senti_test.write("\n")
                            file_create = open(
                                "./ROUGE/models/test" + str(index + index_num * self.batch_size) + ".txt", "w")
                            file_create.writelines(output)
                            file_create.close()

                            file.write("     - train headline: \n")
                            prediction_train_single = prediction_train[index]
                            answer = [self.one_hot[i] for i in prediction_train_single]
                            output = "        "
                            for i in answer:
                                if i != 'UNK' and i != '_PAD':
                                    output += i
                                    output += " "
                            file.write(output)
                            file.write("\n")

                            file.write("     - targets: \n")
                            targets_single = targets[index]
                            answer = [self.one_hot[i] for i in targets_single]
                            output = "        "
                            for i in answer:
                                if i != 'UNK' and i != '_PAD':
                                    output += i
                                    output += " "
                            file.write(output)
                            file.write("\n")
                            print("output %d finished" % (index + index_num * self.batch_size))

                        file.close()
                        # file_senti_test.close()
            else:
                print("model restored failed")
                pass
