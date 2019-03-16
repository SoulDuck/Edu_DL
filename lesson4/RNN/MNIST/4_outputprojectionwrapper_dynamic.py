import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np

# download MNIST Data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""
구현 목적:  
rnn 을 low level api 형태로 코딩해서 원리를 파악합니다.
"""
n_classes = 10
timestep = 28
n_inputs = 28
n_units = 100
lr = 0.001
activation = tf.nn.tanh

tf.reset_default_graph()
# define Input
x = tf.placeholder(shape=[None, timestep, n_inputs], dtype=tf.float32)
y = tf.placeholder(shape=[None, n_classes], dtype=tf.float32)

# Model
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_units)
wrapped_cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size = n_classes)
outputs, states = tf.nn.dynamic_rnn(wrapped_cell, inputs=x, dtype=tf.float32)

""" 
dynamic_rnn cell 을 넣으면 [None, time_step , n_clases] shape 의 tensor 가 나옵니다. 
여기서 우리는 output tensor 을 [:, -1, :] 같이 슬라이스 합니다.
"""
#

logits = outputs[:,-1,:]

# Loss , Optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, logits))
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# Metric
logits_cls = tf.argmax(logits, axis=1)
y_cls = tf.argmax(y, axis=1)
acc = tf.reduce_mean(tf.cast(tf.equal(logits_cls, y_cls), tf.float32))

# Sesion
sess = tf.Session()
sess.run(tf.global_variables_initializer())



# Training
start_time = time.time()
max_step = 50000
for i in range(max_step):
    batch_size = 60
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape(batch_size, 28, 28)
    train_loss, train_acc, _ = sess.run([loss, acc, train_op],
                                        feed_dict={x: batch_xs, y: batch_ys})

    if i % 1000 == 0:
        val_imgs, val_labs = mnist.validation.images, mnist.validation.labels
        n_val = len(val_labs)
        val_imgs = val_imgs.reshape(n_val, 28, 28)
        init_hidden_value = np.zeros(shape=[n_val, n_units], dtype=np.float32)
        val_acc, val_loss = sess.run([acc, loss],
                                     feed_dict={x: val_imgs, y: val_labs})
        print('training acc {:4f} loss {:4f} Validation acc {:4f} , loss {:4f}'. \
              format(train_acc, train_loss, val_acc, val_loss))

# Validation
consume_time = time.time() - start_time
print('batch_size : {} , total step : {} , comsume time : {:4}'. \
      format(batch_size, max_step, consume_time))