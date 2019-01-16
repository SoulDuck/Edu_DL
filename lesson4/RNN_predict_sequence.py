import math
import numpy as np
from numpy import random as npr
import matplotlib.pyplot as plt
import tensorflow as tf


def create_dataset(min_value = 0 , max_value = 30):

    xs = np.arange(min_value, max_value, 0.01)
    ys = []
    for x in xs:
        y = x * math.sin(x) + 2 * math.sin(5 * x) + npr.normal()
        ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def next_batch(ys, n_steps, batch_size):
    n_samples = len(ys)
    xs_seq = []
    ys_seq = []
    for i in range(batch_size):
        ind = npr.randint(0, n_samples - n_steps - 1)
        x_seq = ys[ind: ind + n_steps]
        y_seq = ys[ind + 1: ind + n_steps + 1]
        xs_seq.append(x_seq)
        ys_seq.append(y_seq)

    return np.asarray(xs_seq), np.asarray(ys_seq)


# create dataset
xs_, ys_ = create_dataset(0,30)
train_ys = ys_[:-300]
test_ys = ys_[-300:]

# param
n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

# draw graph
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs- Y))
training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
n_iterations = 30000

batch_size = 60
for iteration in range(n_iterations):
    X_batch, Y_batch = next_batch(ys_, 20, batch_size)
    X_batch = X_batch.reshape([batch_size, n_steps, n_inputs])
    Y_batch = Y_batch.reshape([batch_size, n_steps, n_outputs])
    _, loss_ = sess.run([training_op, loss], feed_dict={X: X_batch, Y: Y_batch})
    print(loss_)


predict_values = []

for i in range(300 - n_steps):

    X_batch = test_ys[i:i + n_steps]
    X_batch = X_batch.reshape([-1, n_steps, n_inputs])
    predict_seq = sess.run(outputs, feed_dict={X: X_batch})
    predict_seq = np.squeeze(predict_seq)
    predict_values.append(predict_seq[-1])


plt.plot(range(300 - n_steps), predict_values, c='r')
plt.plot(range(300), test_ys, c='b')
plt.show()















