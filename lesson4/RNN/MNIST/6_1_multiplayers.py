import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np

# download MNIST Data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""
Multilayer 을 구축합니다. 
구현 조건 
1. RNN 층을 Multilayer 층을 4개 만들고
2. 첫번째 층에는 dropout state 층에 keep prob 을 0.3 줍니다. (n_untis = 100 ) 
3. 2번째 층에는 dropout input 층에 Dropout 을 적용하고 keep prob 을 0.3 줍니다. 
3. 3번째 층에는 OutputProjectionWrapper 을 사용해 logits 을 생성합니다.

dynamic RNN 으로 구현합니다.
 
"""

n_classes = 10
timestep = 28
n_inputs = 28
n_units_0 = 100
n_units_1 = 50
n_units_2 = 50

lr = 0.001
activation = tf.nn.tanh

tf.reset_default_graph()
# define Input
x = tf.placeholder(shape=[None, timestep, n_inputs], dtype=tf.float32)
y = tf.placeholder(shape=[None, n_classes], dtype=tf.float32)
keep_prob = tf.placeholder_with_default(input=0.3, shape=[])

# Model
# Define cell 1
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_units_0)
wrapper_cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=keep_prob)

cell_1 = tf.nn.rnn_cell.BasicRNNCell(num_units=n_units_1)
wrapper_cell_1 = tf.nn.rnn_cell.DropoutWrapper(cell_1, input_keep_prob=keep_prob)

cell_2 = tf.nn.rnn_cell.BasicRNNCell(num_units=n_units_2)
wrapper_cell_2 = tf.contrib.rnn.OutputProjectionWrapper(cell_2, n_classes)

mul_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[wrapper_cell, wrapper_cell_1, wrapper_cell_2])

outputs, states = tf.nn.dynamic_rnn(mul_cell , inputs=x, dtype=tf.float32)

logits = outputs[:, -1, :]

# Loss , Optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
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
        val_acc, val_loss = sess.run([acc, loss],
                                     feed_dict={x: val_imgs, y: val_labs , keep_prob: 1.0})
        print('training acc {:4f} loss {:4f} Validation acc {:4f} , loss {:4f}'.format(train_acc, train_loss, val_acc,
                                                                                       val_loss))

# Validation
consume_time = time.time() - start_time
print('batch_size : {} , total step : {} , comsume time : {:4}'.format(batch_size, max_step, consume_time))