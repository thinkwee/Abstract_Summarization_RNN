# with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
#     encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.encoder_hidden_units, name='encoder_cell')
#
#     self.encoder_output, self.encoder_final_state = tf.nn.dynamic_rnn(
#         cell=encoder_cell,
#         inputs=self.encoder_inputs_embedded,
#         dtype=tf.float32,
#         time_major=False
#     )
# print(self.encoder_final_state)