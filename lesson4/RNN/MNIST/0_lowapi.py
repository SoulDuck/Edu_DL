import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
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
activation = tf.nn.tanh

x = tf.placeholder(shape=[None, timestep, n_inputs] , dtype=tf.float32)
y = tf.placeholder(shape=[None, n_classes], dtype=tf.int64)
init_hidden = tf.placeholder(shape=[None, n_outputs], dtype=tf.float32)
x_trpose = tf.transpose(x, perm=(1, 0, 2))
xs_seq = tf.unstack(x_trpose)

# Model Define

hidden_state_seq = []
# W_in, B_in
random_init = tf.random_normal(shape=[n_inputs, n_outputs], dtype=tf.float32, stddev=0.1)
w_in = tf.Variable(random_init)
b_in = tf.Variable(tf.constant(value=0, shape=[n_outputs], dtype=tf.float32))

# W_hidden, B_hidden
random_init = tf.random_normal(shape=[n_outputs, n_outputs], dtype=tf.float32, stddev=0.1)
w_hidden = tf.Variable(random_init)
b_hidden = tf.Variable(tf.constant(value=0, shape=[n_outputs], dtype=tf.float32))

hidden_state = tf.zeros_like(init_hidden, dtype=tf.float32)

output_layers = []
for i,x_seq in enumerate(xs_seq):
    # inputs
    hidden_layer = tf.matmul(hidden_state , w_hidden)

    now_state = tf.matmul(x_seq, w_in) + b_in
    output_layer = activation(hidden_layer + now_state + b_hidden)
    output_layers.append(output_layer)
    hidden_state = output_layer

logits = tf.layers.dense(output_layers[-1], n_classes)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits , labels=y))
logits_cls = tf.argmax(logits, axis=1)
y_cls = tf.argmax(y, axis=1)
acc = tf.reduce_mean(tf.cast(tf.equal(logits_cls, y_cls), tf.float32))
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
start_time = time.time()
max_step = 50000
for i in range(max_step):
    batch_size = 60
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape(batch_size, 28, 28)
    init_hidden_value = np.zeros(shape=[batch_size, n_outputs], dtype=np.float32)
    train_acc, train_loss, _ = sess.run([acc, loss, train_op],
                                        feed_dict={x: batch_xs, y: batch_ys, init_hidden: init_hidden_value})
    if i % 1000 == 0 :
        val_imgs, val_labs = mnist.validation.images, mnist.validation.labels
        n_val = len(val_labs)
        val_imgs = val_imgs.reshape(n_val, 28, 28)
        init_hidden_value = np.zeros(shape=[n_val, n_outputs], dtype=np.float32)
        val_acc, val_loss = sess.run([acc, loss],
                                        feed_dict={x: val_imgs, y: val_labs, init_hidden: init_hidden_value})
        print('training acc {:4f} , loss {:4f} Validation acc {:4f} , loss {:4f})'.format(train_acc, train_loss, val_acc, val_loss))

consume_time = time.time() - start_time
print('batch_size : {} , total step : {} , comsume time : {:4}'.format(batch_size, max_step, consume_time))