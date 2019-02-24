import tensorflow as tf
import numpy as np
"입력 길이가 가변적인 상황에서 dynamic rnn 을 사용해본다 "

n_inputs = 3
n_steps = 2

x = tf.placeholder(tf.float32 , [None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])  # [2, 1, 2, 2]
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=5)
n_layers = 3
layers = [tf.nn.rnn_cell.BasicRNNCell(num_units=100) for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers , state_is_tuple=False)
outputs , state = tf.nn.dynamic_rnn(multi_layer_cell, x, dtype=tf.float32, sequence_length=seq_length)

print(outputs.shape, state.shape)

# state_is_tuple=False
# Multi layer 을 사용하는 것과 사용하지 않은것과 state 와 outputs의 결과는 차이가 없다
# outputs.shape =>  shape=(?, 2, 100)
# state.shape => shape=(?, 100)

# *** 여기가 중요한 핵심 포인트 ***
# state_is_tuple=True
# 각 layer 의 모든 state 을 순서대로 담겨져 있다
# outputs.shape =>  shape=(?, 2, 100)
# state.shape => shape=(?, **300**)
# 왜 state 의 상태를 모두 담는 행위를 했을까




############################
# time_step1, time_step2   #
# [[0, 1, 2], [9, 8, 7]]   #
############################
x_batch = np.array([
    [[0, 1, 2], [9, 8, 7]],
    [[3, 4, 5], [0, 0, 0]], # > 비어 있는 sequence 는 0,0,0 으로 둔다
    [[6, 7, 8], [6, 5, 4]],
    [[9, 0, 1], [3, 2, 1]]
])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

outputs_, state_ = sess.run(fetches=[outputs, state], feed_dict={x: x_batch, seq_length:[2,1,2,2]})
print(outputs_)
print(outputs_.shape)
print(state_) # 각 셀의 0 백터가 제외된 마지막 상태를 담습니다
print(state_.shape) #





