import tensorflow as tf
var = tf.Variable('hello world')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(var))