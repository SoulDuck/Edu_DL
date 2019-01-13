import tensorflow as tf
saver = tf.train.import_meta_graph('./model/mandelbrot.meta')
sess = tf.Session()
saver.restore(sess, './model/mandelbrot')
ns=tf.get_default_graph().get_tensor_by_name('cal/ns:0')
step = tf.get_default_graph().get_operation_by_name('step')
ns_ = sess.run('ns:0')
sess.run('step')


import matplotlib.pyplot as plt
plt.imshow(ns_)
plt.show()
