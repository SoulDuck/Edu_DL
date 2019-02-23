import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
"""

"""


# Data
def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t * 5)


t_min, t_max = 0, 30
resolution = 0.1

# Training Data
n_steps = 21
n_inputs = 1
n_neurons = 100
n_outputs = 1

start_train = 0.0
end_train = 12.0
train_x_axis = np.arange(start_train, end_train, 0.1)
train_dataset = time_series(train_x_axis)

# Validation Data
start_val = 12.0
end_val = 30.0
val_x_axis = np.arange(start_val, end_val, 0.1)
val_dataset = time_series(val_x_axis)


# Make dataset
def generate_predict(datum, timestep):
    xs = []
    ys = []
    for i in range(len(datum) - timestep):
        """
        please first read this 
        if x is =[ 0 ,0.1, 0.2, 0.3, 0.4 , 0.5], timestpe is 5 
        x = [0 ,0.1, 0.2, 0.3, 0.4] <- datum[0: 0 + timestep
        y = [0.5] <- datum[0: 0 + timestep <- datum[i + timestep]
        """
        xs.append(datum[i: i + timestep])  # if time step = 5 [0 ,0.1, 0.2, 0.3, 0.4]
        ys.append(datum[i + timestep])  # if time step = 5[0.5]
    return np.asarray(xs), np.asarray(ys)


train_xs, train_ys = generate_predict(train_dataset, n_steps)
val_xs, val_ys = generate_predict(val_dataset, n_steps)


def next_batch(xs, ys, batch_size):
    indices = np.random.choice(len(xs), batch_size, replace=True)
    return xs[indices], ys[indices]


x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None])

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
stacked_outputs = tf.reshape(outputs, shape=[-1, n_neurons])
stacked_logits = tf.layers.dense(stacked_outputs, n_outputs)
outputs = tf.reshape(stacked_logits, [-1, n_steps, n_outputs])

lr = 0.01
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train parameter
iterations = 2000
batch_size = 60

# Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(iterations):
    batch_xs, batch_ys = next_batch(train_xs, train_ys, batch_size)
    print(batch_xs.shape)
    batch_xs = batch_xs.reshape(batch_size, n_steps, n_inputs)
    _, loss_ = sess.run([train_op, loss], feed_dict={x: batch_xs, y: batch_ys})
    print(loss_)

# create sequence
predicts = []

init_xs = np.zeros(shape=[1, n_steps, n_inputs], dtype=np.float32)
x_ = init_xs
ys = []
for _ in range(100):
    outputs_, states_ = sess.run([outputs, states], feed_dict={x: x_})
    pred = np.squeeze(outputs_)[-1]
    x_ = np.roll(x_, -1)
    x_[-1] = pred
    ys.append(pred)

# Create
# visualization Validation Dataset
ys = np.hstack([np.zeros([4 + n_steps - 1]) , ys])
x_axis = range(len(ys))
plt.scatter(x_axis, ys, c='r', label='true')
#plt.scatter(x_axis, predicts, c='b', label='predict')
#plt.legend()
plt.show()