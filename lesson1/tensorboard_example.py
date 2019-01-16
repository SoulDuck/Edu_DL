import tensorflow as tf

init = tf.random_normal(shape=[520,600], name='rand_norm')
A = tf.Variable(init)
A_mean = tf.reduce_mean(A)
tf.summary.histogram(name='histogram',values = A)
merge_all = tf.summary.merge_all()
print(merge_all)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merge_all_ = sess.run(merge_all)




