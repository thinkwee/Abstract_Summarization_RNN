import tensorflow as tf

a = [[2.0, 5.0, 4.0, 1.0, 2.0], [1.0, 2.0, 3.0, 5.0, 4.0]]
b = tf.nn.batch_normalization(a, [2.8, 3.0], [0.2, 0.3], offset=[0.01, 0.01], scale=[0.1, 0.2], variance_epsilon=0.001)

with tf.Session() as sess:
    bn = sess.run(b)
    print(bn)
