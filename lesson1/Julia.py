import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Concept
# input initial value
#Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Y, X = np.mgrid[-2:2:0.005, -2:2:0.005]

init_value = X + 1j * Y

c = np.complex(0.0,0.75)
c = tf.constant(c)

xs = tf.constant(init_value)
zs = tf.Variable(xs)
zs_zeros = tf.zeros_like(xs, tf.float32)
ns = tf.Variable(zs_zeros)

zs_squre = tf.multiply(zs, zs)
zs_sub = tf.subtract(zs_squre, c)
zs_abs = tf.abs(zs_sub)

zs_less = tf.math.less(zs_abs, 4)
zs_cast = tf.cast(zs_less , tf.float32)

#
step = tf.group(
  tf.assign(zs, zs_sub),
  tf.assign_add(ns, zs_cast)
)

#
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(200):
    sess.run(step)
value = sess.run(ns)
plt.imshow(value)
plt.show()