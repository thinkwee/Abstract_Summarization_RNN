self.decoder_logits_test = tf.contrib.layers.linear(self.decoder_outputs_test, self.vocab_size)
self.decoder_prediction = tf.argmax(self.decoder_logits_test, 2)

self.targets = tf.reshape(self.decoder_targets, [-1])
self.logits_flat = tf.reshape(self.decoder_train_logits.rnn_output, [-1, self.vocab_size])





self.logits_flat = tf.argmax(self.logits_flat, 0)

self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=tf.one_hot(self.decoder_targets,
                      depth=self.vocab_size,
                      dtype=tf.float32),
    logits=self.model_outputs,
)
self.loss = tf.reduce_mean(self.stepwise_cross_entropy)

self.loss = tf.losses.sparse_softmax_cross_entropy(self.targets, self.logits_flat)


