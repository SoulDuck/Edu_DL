import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_classes = 10
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')
phase_train = tf.placeholder(dtype=tf.bool , shape=[], name='phase_train')
lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

activation = tf.nn.relu
layer_0 = tf.layers.dense(x, units=128, activation=activation)
layer_0 = tf.layers.dropout(layer_0, rate=0.5, training=phase_train)

layer_1 = tf.layers.dense(layer_0, units=256, activation=activation)
layer_1 = tf.layers.dropout(layer_1, rate=0.5, training=phase_train)

logits = tf.layers.dense(layer_1, units=10, activation=activation)
loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())



vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for i in range(10):
    batch_xs, batch_ys = mnist.train.next_batch(60)
    _, loss_ = sess.run([train_op, loss], feed_dict={x: batch_xs, y: batch_ys, phase_train: False, lr: 0.001})
    vars_ = sess.run(vars , feed_dict={x: batch_xs, y: batch_ys, phase_train: False, lr: 0.001})
    if i % 10 == 0:
        var_hists = list(map(np.histogram, vars_))
        plt.bar(range(len(var_hists[0][0])), var_hists[0][0])
        plt.show()


