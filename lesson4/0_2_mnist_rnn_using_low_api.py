import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""
Tip: 
모든 열에 대한 정보가 담겨져있는 states 와 Fully connected layer 을 연결합니다
"""

n_classes = 10
timestep = 28
n_inputs = 28
n_outputs = 100
lr = 0.001
activation=tf.nn.tanh


x = tf.placeholder(shape=[None , timestep , n_inputs] , dtype=tf.float32)
y = tf.placeholder(shape=[None , n_classes] , dtype=tf.float32)
init_hidden = tf.placeholder(shape=[None ,n_outputs] , dtype=tf.float32)
x_trpose = tf.transpose(x , perm=(1,0,2))
xs_seq = tf.unstack(x_trpose)


# Input Layer
hidden_state_seq = []

# W_in, B_in
random_init = tf.random_normal(shape=[n_inputs, n_outputs], dtype=tf.float32, stddev=0.1)
w_in = tf.Variable(random_init)
b_in = tf.Variable(tf.constant(value=0, shape=[n_outputs], dtype=tf.float32))

# W_hidden, B_hidden
random_init = tf.random_normal(shape=[n_outputs, n_outputs], dtype=tf.float32, stddev=0.1)
w_hidden = tf.Variable(random_init)
b_hidden = tf.Variable(tf.constant(value=0, shape=[n_outputs], dtype=tf.float32))

# tf.constant을 사용할수 없다 1차원에 None 이 들어가기 때문인데 그래서 zeros_like 을 사용하는 방법으로 workaround 로 피해갔는데
# 다른 방법이 없을까?
hidden_state = tf.zeros_like(init_hidden, dtype=tf.float32)

output_layers = []
for i,x_seq in enumerate(xs_seq):
    # inputs
    hidden_layer = tf.matmul(hidden_state , w_hidden)

    now_state = tf.matmul(x_seq, w_in) + b_in
    output_layer = activation(hidden_layer  + now_state + b_hidden)
    output_layers.append(output_layer)
    hidden_state = output_layer

logits = tf.layers.dense(output_layers[-1],n_classes)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits , labels=y))
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3000):
    batch_size = 60
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape(batch_size, 28,28)
    init_hidden_value = np.zeros(shape=[batch_size, n_outputs], dtype=np.float32)
    _, train_loss = sess.run([train_op, loss], feed_dict={x: batch_xs , y: batch_ys , init_hidden: init_hidden_value})
    print(train_loss)
