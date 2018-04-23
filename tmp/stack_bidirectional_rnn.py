cell_bw_list = []
cell_fw_list = []
for _ in range(self.num_layers):
    cell_fw, cell_bw = self._create_bgrucell()
    cell_fw_list.append(cell_fw)
    cell_bw_list.append(cell_bw)
(output, encoder_final_state_fw, encoder_final_state_bw) = rnn.stack_bidirectional_rnn(
    cells_fw=cell_fw_list,
    cells_bw=cell_bw_list,
    inputs=inputs,
    dtype=tf.float32,
    sequence_length=self.encoder_length)
self.encoder_final_state = tf.concat(axis=1, values=[encoder_final_state_fw[self.num_layers - 1],
                                                     encoder_final_state_bw[self.num_layers - 1]])