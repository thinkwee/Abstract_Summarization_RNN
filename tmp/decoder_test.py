self.decoder_outputs_train, _ = tf.nn.dynamic_rnn(
    cell=decoder_cell,
    inputs=self.decoder_inputs_embedded,
    initial_state=self.encoder_final_state,
    dtype=tf.float32,
    sequence_length=self.decoder_length,
    time_major=False)

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
decoder_infer_outputs, _, _ = s2s.dynamic_decode(
    decoder, maximum_iterations=25)
self.decoder_prediction = decoder_infer_outputs.sample_id
