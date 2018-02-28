import tensorflow as tf

global_step = tf.Variable(0, trainable=False)

initial_learning_rate = 0.005

learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=10, decay_rate=0.2)
opt = tf.train.GradientDescentOptimizer(learning_rate)

add_global = global_step.assign_add(1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))
    for i in range(30):
        _, rate = sess.run([add_global, learning_rate])
        print("%9.9f" % rate)
