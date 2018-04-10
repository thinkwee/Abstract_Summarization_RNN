import numpy as np
import tensorflow as tf

CLASS = 10
label1 = tf.constant([5, 2, 1])
sess1 = tf.Session()
print('label1:', sess1.run(label1))
b = tf.one_hot(label1, depth=CLASS, dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(b)
    print('after one_hot', sess.run(b))
