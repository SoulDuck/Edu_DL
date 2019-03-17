import tensorflow as tf
import numpy as np

tf.reset_default_graph()
print(tf.__version__)

"""
multi-layer 
Cell 을 여러층을 쌓아 올립니다.
"""
n_inputs = 3
n_outputs = 1
n_steps = 2
n_units = 5
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])  # [2, 1, 2, 2]
keep_prob = tf.placeholder_with_default(input=0.001, shape=[])  # [2, 1, 2, 2]

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_units)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=1.0,
                                     state_keep_prob=keep_prob)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=1)


outputs, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)  # , sequence_length=seq_length

stacked_ouput = tf.reshape(outputs, [-1, n_units])
stacked_layer = tf.layers.dense(stacked_ouput, n_outputs)
logits = tf.reshape(stacked_layer, [-1, n_steps, n_outputs])

############################
# time_step1, time_step2   #
# [[0, 1, 2], [9, 8, 7]]   #
############################
x_batch = np.array([
    [[0, 1, 2], [9, 8, 7]],
    [[3, 4, 5], [0, 0, 0]],  # > 비어 있는 sequence 는 0,0,0 으로 둔다
    [[6, 7, 8], [6, 5, 4]],
    [[9, 0, 1], [3, 2, 1]]
])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

outputs_, state_ = sess.run(fetches=[outputs, state], feed_dict={x: x_batch, seq_length: [2, 1, 2, 2]})

print(outputs_)
print(outputs_.shape)
print(state_)