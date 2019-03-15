import tensorflow as tf
import numpy as np
"입력 길이가 가변적인 상황에서 dynamic rnn 을 사용해본다 "
"""
1.
tf.nn.dynamic_rnn  을 사용하는 이유는 staticrnn 은 정방향 패스에 계산된 모든 텐서를 저장해서 역방향 패스의
그래디언트를 계산할때 사용하기 때문이다

2.dynamic_rnn 은 while_loop 을 이용한다.
 그리고 타입스텝의 output 을 모아 하나의 함수로 반환한다. 
 output shape = [None, n_steps, n_neurons]
 
3. 가변적인 길이를 담을때는 seq_length 을 설정한다 
그러면 0백터는 제외한 state 가 담긴다
물론 outputs 에는 모든게 담긴다
 
"""




n_inputs = 3
n_steps = 2

x = tf.placeholder(tf.float32 , [None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])  # [2, 1, 2, 2]
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=5)
outputs , state = tf.nn.dynamic_rnn(cell , x, dtype=tf.float32,  sequence_length=seq_length)
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

outputs_, state_ = sess.run(fetches=[outputs, state], feed_dict={x: x_batch, seq_length:[2, 1, 2, 2]})

print(outputs_)
print(outputs_.shape)
print(state_) # 각 셀의 0 백터가 제외된 마지막 상태를 담습니다
print(state_.shape) #





