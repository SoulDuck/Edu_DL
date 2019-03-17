import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t * 5)


def normalize(min_, max_, datum):
    datum = np.asarray(datum)
    return (datum - min_)/(max_ - min_)


def generate_predict(datum, timestep):
    sliced_xs = []
    sliced_ys = []
    for ind in range(len(datum) - timestep):
        """
        please first read this 
        if x is =[ 0 ,0.1, 0.2, 0.3, 0.4 , 0.5], timestpe is 5 
        x = [0 ,0.1, 0.2, 0.3, 0.4] <- datum[0: 0 + timestep
        y = [0.5] <- datum[0: 0 + timestep <- datum[i + timestep]
        """
        sliced_xs.append(datum[ind: ind + timestep])  # if time step = 5 [0 ,0.1, 0.2, 0.3, 0.4]
        sliced_ys.append(datum[ind + 1: ind + timestep + 1])  # if time step = 5[0.5]
    return np.asarray(sliced_xs), np.asarray(sliced_ys)


def next_batch(xs, ys, n_batch):
    indices = np.random.choice(range(len(xs)), n_batch, replace=False)
    return xs[indices], ys[indices]


# 데이터의 범위를 설정합니다.
t_min, t_max = 0, 30
resolution = 0.1

# Training Data
start_train = 0.0
end_train = 25.0  # 12.2 + 0.1 * (n_steps + 1)
train_x_axis = np.arange(start_train, end_train, 0.1)
# train_x_axis = normalize(np.min(train_x_axis) , np.max(train_x_axis), train_x_axis)
train_dataset = time_series(train_x_axis)

# Validation Data
start_val = 25.0
end_val = 30.0
val_x_axis = np.arange(start_val, end_val, 0.1)
# val_x_axis = normalize(np.min(train_x_axis) , np.max(train_x_axis), train_x_axis)
val_dataset = time_series(val_x_axis)

# Time Step 의 길이를 지정합니다
n_steps = 21
train_xs, train_ys = generate_predict(train_dataset, n_steps)
val_xs, val_ys = generate_predict(val_dataset, n_steps)

# Model Parameter
n_inputs = 1
n_neurons = 100
n_outputs = 1
lr = 0.001

# define Input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
keep_prob = tf.placeholder_with_default(0.7, shape=[])

# Model
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
wrapped_cell = tf.nn.rnn_cell.DropoutWrapper(cell , state_keep_prob=keep_prob)
wrapped_cell = tf.contrib.rnn.OutputProjectionWrapper(wrapped_cell, n_outputs)

outputs, states = tf.nn.dynamic_rnn(wrapped_cell, inputs=x, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train parameter
iterations = 2000
batch_size = 60

# Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# Loss
batch_xs, batch_ys = next_batch(train_xs, train_ys, batch_size)
batch_xs, batch_ys = [batch.reshape([batch_size, n_steps, n_inputs]) for batch in [batch_xs, batch_ys]]


for i in range(iterations):
    batch_xs, batch_ys = next_batch(train_xs, train_ys, batch_size)
    batch_xs, batch_ys = [batch.reshape([batch_size, n_steps, n_inputs]) for batch in [batch_xs, batch_ys]]
    _, loss_ = sess.run([train_op, loss], feed_dict={x: batch_xs, y: batch_ys})
    print(loss_)

# Eval
predicts = []
for val_x in val_xs:
    val_x = val_x.reshape(1, n_steps, n_inputs)
    outputs_ = sess.run([outputs], feed_dict={x: val_x, keep_prob:1.7})
    predicts.append(np.squeeze(outputs_)[-1])

# visualization Validation Dataset
# x 축을 값을 가져옵니다 , 그리고 정답과 예측값을 동시에 비교합니다
x_axis = val_x_axis[n_steps:]
plt.scatter(x_axis, val_dataset[n_steps:], c='r', label='true')
plt.scatter(x_axis, predicts, c='b', label='predict')
plt.legend()
plt.show()