import tensorflow as tf

y_ = tf.constant([1, 2, 3], dtype=tf.float32, shape=[3, 1])
y1 = tf.constant([1, 2, 3], dtype=tf.float32, shape=[3, 1])
y2 = tf.constant([8, 4, 1], dtype=tf.float32, shape=[3, 1])

MSE1 = tf.reduce_mean(tf.square(y1 - y_))
MSE2 = tf.reduce_mean(tf.square(y2 - y_))

with tf.Session() as sess:
    print(sess.run(MSE1))
    print(sess.run(MSE2))
