import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_classes = 10
timestep = 28
lr = 0.001
x = tf.placeholder(shape=[None , timestep , 28] , dtype=tf.float32)
y = tf.placeholder(shape=[None , n_classes] , dtype=tf.float32)
x_trpose = tf.transpose(x , perm=(1,0,2))
x_seq = tf.unstack(x_trpose)
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=100)
outputs , hidden = tf.nn.static_rnn(cell, inputs=x_seq, dtype=tf.float32)

init_value_W= tf.random_normal([100, n_classes], dtype=tf.float32)
init_value_B = tf.random_normal([n_classes], dtype=tf.float32)
W=tf.Variable(init_value_W)
B=tf.Variable(init_value_B)
output = tf.matmul(hidden ,W) +B

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3000):
    batch_size = 60
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape(batch_size, 28,28)
    _, train_loss = sess.run([train_op, loss], feed_dict={x: batch_xs , y: batch_ys})
    print(train_loss)

