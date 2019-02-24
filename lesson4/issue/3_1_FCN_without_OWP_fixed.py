import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Issue : 
y 을 
y = tf.placeholder(tf.float32, [None, n_outputs]) 이렇게 하면 학습이 되지 않는다
반면 y = tf.placeholder(tf.float32, [None, timesteps, inputs]) 이렇게 하면 학습이 된다

output = (batch_size, 21, 1) 의 마지막 timestep 을 가져온다
pred_value = squeeze(output)[:,-1]
loss 는 square(y - pred_value )**2
  
왜 학습이 안될까 
내가 추론하기로 이유는 

timestep1 timestep2 timestep3 ... timestep21 이 있는데
y = tf.placeholder(tf.float32, [None, n_outputs]) 을 이용하면 time step1 부터 21 까지 모두 loss 가 나와서 
학습을 하는 방면 , 마지막만 주면 학습이 time step 21 만 학습이 되기 때문이라 고 생각한다
 
아니다 코드가 걍 잘 못되었다 

어디가 문제 있었나
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

start_train = 12.2
end_train = 18.0  # 12.2 + 0.1 * (n_steps + 1)
train_x_axis = np.arange(start_train, end_train, 0.1)
train_dataset = time_series(train_x_axis)

# Validation Data
val_x_axis = np.hstack([np.arange(0, start_train, 0.1), np.arange(end_train, 30, 0.1)])
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


# Model
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
stacked_outputs = tf.reshape(outputs, shape=[-1, n_neurons])
stacked_logits = tf.layers.dense(stacked_outputs, n_outputs)
outputs = tf.reshape(stacked_logits, [-1, n_steps, n_outputs])
pred_value = outputs[:, :, 0]
pred_value = pred_value[:, -1]

lr = 0.001
loss = tf.reduce_mean(tf.square(y - pred_value))
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train parameter
iterations = 1000
batch_size = 60

# Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(iterations):
    batch_xs, batch_ys = next_batch(train_xs, train_ys, batch_size)
    batch_xs = batch_xs.reshape(batch_size, n_steps, n_inputs)

    _, loss_ = sess.run([train_op, loss], feed_dict={x: batch_xs, y: batch_ys})
    print(loss_)

# Eval
predicts = []
for xs in val_xs:
    xs = xs.reshape(1, n_steps, n_inputs)
    outputs_, states_ = sess.run([outputs, states], feed_dict={x: xs})
    predicts.append(np.squeeze(outputs_)[-1])

# visualization Validation Dataset
x_axis = val_x_axis[(n_steps - 1): (n_steps - 1) + len(val_ys)]
plt.scatter(x_axis, val_ys, c='r', label='true')
plt.scatter(x_axis, predicts, c='b', label='predict')
plt.legend()
plt.show()



