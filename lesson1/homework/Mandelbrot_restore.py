import tensorflow as tf
saver = tf.train.import_meta_graph('./model/mandelbrot.meta')
sess = tf.Session()
saver.restore(sess, './model/mandelbrot')
ns=tf.get_default_graph().get_tensor_by_name('ns:0')
ns_ = sess.run(ns)

import matplotlib.pyplot as plt
plt.imshow(ns_)
plt.show()
