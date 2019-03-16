import tensorflow as tf
import numpy as np
print(tf.__version__)

"""
마지막 output layer 에 fully connected layer 을 연결하고 싶을때는 
 tf.contrib.OutputProjectionWrapper 을 사용합니다
  
  해석 : Projection 이라는 뜻을 생각해 보면 Semantic 과 유사하다는 것을 알수 있습니다 
  Projecter 을 생각해보면 작은 점에서 큰 영상이 나옵니다 그것과 유사합니다 
  Output 을 작은 몇개의 점으로 Embedding 하는 것을 보통 Projection 한다고 하고
  그 몇개의 점은 Semantic 한 특성이 있다 라고 말할수 있습니다 
 
 Output 을 Projection 하는 2가지 방법 
 1. OutputProjectionWrapper 을 사용한다.
 2. 직접 Fully Connected Layer 을 연결합니다. 출력된 Output 을 모아 stack 한 후 Projection 합니다.
  
  
  Issue : 왜 안되는지 모르겠다.
  colab version 1.13 에서는 돌아간다  
"""

n_inputs = 3
n_steps = 2

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])  # [2, 1, 2, 2]

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=5)
wrapped_cell = tf.contrib.rnn.OutputProjectionWrapper(cell=cell, output_size=1,)
outputs, state = tf.nn.dynamic_rnn(wrapped_cell , x, dtype=tf.float32, sequence_length=seq_length)
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
print(state_)  # 각 셀의 0 백터가 제외된 마지막 상태를 담습니다
print(state_.shape)  #