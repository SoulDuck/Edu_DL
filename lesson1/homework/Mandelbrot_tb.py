import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# add tensorboard
# input initial value
# Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
init_value = X + 1j*Y
#

with tf.name_scope('cal'):
    xs = tf.constant(init_value )
    zs = tf.Variable(xs)
    zs_zeros = tf.zeros_like(xs, tf.float32)
    ns = tf.Variable(zs_zeros,name='ns')
    print(ns)


ns_image_tb = tf.summary.image(name='ns_image',tensor=tf.reshape(ns,shape=[1,520,600,1]))

zs_squre = tf.multiply(zs,zs)
zs_add = tf.add(zs_squre , xs)
zs_abs = tf.abs(zs_add)
zs_less = tf.math.less(zs_abs , 4)
zs_cast = tf.cast(zs_less , tf.float32)

cast_mean_tb = tf.summary.scalar(name='cast_mean',tensor=tf.reduce_mean(zs_cast))
cast_hist_tb = tf.summary.histogram(name='cast_hist',values=zs_cast )
#
step = tf.group(
  tf.assign(zs, zs_add),
  tf.assign_add(ns, zs_cast)
)

#
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
tbs = tf.summary.merge_all()

writer = tf.summary.FileWriter(logdir = './tensorboard')
writer.add_graph(tf.get_default_graph())

for i in range(200):
    _, tbs_,cast_ = sess.run([step,tbs,zs_cast])
    print(cast_)
    writer.add_summary(tbs_,global_step=i)

saver.save(sess,save_path='./model/mandelbrot')
value = sess.run(ns)
plt.imshow(value)
plt.show()