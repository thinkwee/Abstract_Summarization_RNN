self._build_graph()
            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
            ckpt = tf.train.get_checkpoint_state(self.MODEL_FILE)
            with tf.Session() as sess:
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print("the model has been successfully restored")
                    for index in range(num_train_steps):
                        encoder_inputs, decoder_inputs, decoder_targets, encoder_length, decoder_length = next(batches)
                        decoder_length = [decoder_length for _ in range(self.batch_size)]
                        encoder_length = [encoder_length for _ in range(self.batch_size)]
                        feed_dict = {
                            self.decoder_targets: decoder_targets,
                            self.decoder_length: decoder_length,
                            self.encoder_inputs: encoder_inputs,
                            self.decoder_inputs: decoder_inputs,
                            self.encoder_length: encoder_length
                        }
                        file = open("./infer/output.txt", "w")
                        loss_infer_total = 0.0
                        for test_index in range(self.batch_size):

                            file.write("- group %d\n" % (test_index + 1))

                            file.write("     - infer headline: \n")
                            logits_infer = sess.run(self.decoder_infer_logits, feed_dict=feed_dict)
                            prediction_infer = logits_infer.sample_id
                            answer = [one_hot[i] for i in prediction_infer[test_index]]
                            output = "        "
                            for i in answer:
                                if i != "UNK":
                                    output += i
                                    output += " "
                            file.write(output)
                            file.write("\n")

                            file.write("     - targets: \n")
                            targets = sess.run(self.decoder_targets, feed_dict=feed_dict)
                            answer = [one_hot[i] for i in targets[test_index]]
                            output = "        "
                            for i in answer:
                                if i != "UNK":
                                    output += i
                                    output += " "
                            file.write(output)
                            file.write("\n")
                            print("output %d finished" % test_index)

                            loss_infer_total += sess.run(self.loss_infer, feed_dict=feed_dict)
                            print(loss_infer_total)

                        file.write("average infer loss: %9.9f" % (loss_infer_total / self.batch_size))

                        file.close()
                        print("infer file updated")
                else:
                    print("model restored failed")
                    pass