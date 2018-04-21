import numpy as np

a = [13, 10, 10, 11, 12, 10, 11, 9, 13, 13, 9, 12, 9, 10, 11, 8, 12, 17, 8, 13, 10, 17, 14, 12, 7, 13, 12, 10, 15, 7,
     10, 14]
print(np.sum(a))

# self.masks_infer = tf.sequence_mask(lengths=self.decoder_length, maxlen=self.decoder_max_iter,
#                                     dtype=tf.float32,
#                                     name='masks_infer')
# self.loss_infer = contrib.seq2seq.sequence_loss(targets=self.targets,
#                                                 logits=self.logits_infer,
#                                                 weights=self.masks_infer)

# self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_targets,
#                                                            logits=self.logits_train)
# self.loss_infer = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_targets,
#                                                                  logits=self.logits_infer)