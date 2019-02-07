import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5
n_steps = 2

X = tf.placeholder(tf.float32 , [None , n_steps, n_inputs])
X_ = tf.transpose(X,[1,0,2])
X_seq = tf.unstack(X_)


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs , states = tf.contrib.rnn.static_rnn(basic_cell, X_seq, dtype=tf.float32)
outputs = tf.transpose(tf.stack(output_seqs), perm=[1,0,2])

sess = tf.Session()
init = tf.global_variables_initializer()

X_batch = np.array([
    [[0,1,2],[9,8,7]],
    [[3,4,5],[0,0,0]],
    [[9,0,1],[3,2,1]],
])

sess.run(init)
outputs__, X__ = sess.run([outputs, X_], feed_dict={X: X_batch})





