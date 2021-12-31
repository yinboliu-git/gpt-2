import tensorflow as tf

a = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
a1 = tf.tile(a, [1, 1])
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(tf.shape(a)))
    print(sess.run(a1))
    print(sess.run(tf.shape(a1)))
