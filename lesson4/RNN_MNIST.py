import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.001

X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
Y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)

loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, Y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

# load MNIST dataset

mnist = input_data.read_data_sets('/tmp/data/')
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
Y_test = mnist.test.labels

n_epochs = 100
batch_size = 150

sess = tf.Session()
sess.run(init)

for epoch in range(n_epochs):
    for iteration in range(mnist.train.num_examples // batch_size):
        X_batch, Y_batch = mnist.train.next_batch(batch_size=150)
        X_batch = X_batch.reshape((-1, n_steps, n_inputs))
        sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})

    train_acc = sess.run(accuracy, feed_dict={X: X_batch, Y: Y_batch})
    test_acc = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})

    print(train_acc)
    print(test_acc)
`