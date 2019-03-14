import tensorflow as tf
import numpy as np
"입력 길이가 가변적인 상황에서 dynamic rnn 을 사용해본다 "

n_inputs = 3
n_steps = 2

x = tf.placeholder(tf.float32 , [None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])  # [2, 1, 2, 2]
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=5)
outputs , state = tf.nn.dynamic_rnn(cell , x , dtype=tf.float32,  sequence_length=seq_length)
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





