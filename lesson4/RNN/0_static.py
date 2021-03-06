import tensorflow as tf
import numpy as np
"입력 길이가 정적인 상황에서 static rnn 을 사용해본다 "
n_inputs = 3
n_steps = 2

x = tf.placeholder(tf.float32 , [None, n_steps, n_inputs])
x_trp = tf.transpose(x , perm=[1, 0, 2])
x_seq = tf.unstack(x_trp)


cell = tf.nn.rnn_cell.BasicRNNCell(num_units=5)
outputs , state = tf.nn.static_rnn(cell, x_seq, dtype=tf.float32)


############################
# time_step1, time_step2   #
# [[0, 1, 2], [9, 8, 7]]   #
############################
x_batch = np.array([
    [[0, 1, 2], [9, 8, 7]],
    [[3, 4, 5], [1, 2, 3]], # > 비어 있는 sequence 는 0,0,0 으로 둔다
    [[6, 7, 8], [6, 5, 4]],
    [[9, 0, 1], [3, 2, 1]]
])


sess = tf.Session()
sess.run(tf.global_variables_initializer())

outputs_, state_ = sess.run(fetches=[outputs, state], feed_dict={x: x_batch})
print(outputs_)
print(state_)

"""
결과 해석

Outputs:                                                                      time step 1 에 대한 결과
[array([[ 0.7044115 , -0.30623248,  0.12037031, -0.85711247, -0.500985 ],   | [0, 1, 2]
       [ 0.9987269 ,  0.736788  ,  0.30991018, -0.9337488 , -0.9914549 ],   | [3, 4, 5]
       [ 0.99999535,  0.97589666,  0.47765014, -0.969947  , -0.99988925],   | [6, 7, 8]
       [ 0.9168477 ,  0.9999249 ,  0.72356266,  0.99986637, -0.9998162 ]],  | [9, 0, 1]
                                                                              time step 2 에 대한 결과()
                                                                              time step 2 여기 결과가 timestep 의 
                                                                              마지막임으로 states 로 보내집니다
 array([[ 0.9999994 ,  0.9985358 ,  0.6736399 , -0.01696424, -0.9999344 ],  | [9, 8, 7]
       [ 0.9670452 ,  0.3959666 , -0.12885226, -0.80039454, -0.08191571],   | [1, 2, 3]
       [ 0.99992615,  0.9975476 , -0.10483205,  0.70091784, -0.9884625 ],   | [6, 5, 4] 
       [ 0.98945767,  0.9967613 , -0.25809166,  0.8049452 , -0.722623  ]],  | [3, 2, 1] 
      dtype=float32)]
      
States: outputs 에 마지막 array 을 보여줌       
[[ 0.9999994   0.9985358   0.6736399  -0.01696424 -0.9999344 ]
 [ 0.9670452   0.3959666  -0.12885226 -0.80039454 -0.08191571]
 [ 0.99992615  0.9975476  -0.10483205  0.70091784 -0.9884625 ]
 [ 0.98945767  0.9967613  -0.25809166  0.8049452  -0.722623  ]]


"""





