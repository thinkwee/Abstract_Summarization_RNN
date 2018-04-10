import tensorflow as tf

batch_size = 2

articles = tf.constant([[1, 2, 4], [2, 3, 9]])
encoder_inputs = tf.unstack(tf.transpose(articles))

with tf.Session() as sess:
    encoder_inputs = (sess.run(encoder_inputs))
    for i in encoder_inputs:
        print(i)
