# # single layer blstm encoder
# for layer_i in range(self.encoder_layers):
#     with tf.variable_scope('encoder%i' % layer_i, reuse=tf.AUTO_REUSE):
#         cell_fw = rnn.LSTMCell(
#             num_units=self.encoder_hidden_units,
#             initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=114),
#             state_is_tuple=True)
#         cell_bw = rnn.LSTMCell(
#             num_units=self.encoder_hidden_units,
#             initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=133),
#             state_is_tuple=True)
#         (self.encoder_inputs_embedded, self.encoder_final_state) = tf.nn.bidirectional_dynamic_rnn(
#             cell_fw=cell_fw,
#             cell_bw=cell_bw,
#             inputs=self.encoder_inputs_embedded,
#             dtype=tf.float32)
# self.encoder_final_state_c = tf.concat(
#     (self.encoder_final_state[0].c, self.encoder_final_state[1].c), 1)
# self.encoder_final_state_h = tf.concat(
#     (self.encoder_final_state[0].h, self.encoder_final_state[1].h), 1)
# self.encoder_final_state = contrib.rnn.LSTMStateTuple(
#     c=self.encoder_final_state_c,
#     h=self.encoder_final_state_h)