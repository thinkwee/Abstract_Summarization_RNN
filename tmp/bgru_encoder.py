# # multi layer bgru encoder with dropout wrapper
#         with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
#
#             inputs = self.encoder_inputs_embedded
#             for layer_i in range(self.encoder_layers):
#                 with tf.variable_scope(None, default_name="bidirectional-rnn-%i" % layer_i):
#                     cell_fw = tf.nn.rnn_cell.GRUCell(
#                         num_units=self.encoder_hidden_units,
#                         kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=114))
#                     cell_bw = tf.nn.rnn_cell.GRUCell(
#                         num_units=self.encoder_hidden_units,
#                         kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=114))
#                     if self.is_train:
#                         cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
#                         cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)
#
#                     (output, self.encoder_final_state) = tf.nn.bidirectional_dynamic_rnn(
#                         cell_fw=cell_fw,
#                         cell_bw=cell_bw,
#                         inputs=inputs,
#                         dtype=tf.float32)
#                     inputs = tf.concat(output, 2)
#             self.encoder_final_state = tf.concat(self.encoder_final_state, 1)