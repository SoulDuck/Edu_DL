import tensorflow as tf
import numpy as np

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t * 5)

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
        ys.append(datum[i + 1: i + timestep + 1])  # if time step = 5[0.5]
    return np.asarray(xs), np.asarray(ys)


def next_batch(xs, ys, batch_size):
    indices = np.random.choice(range(len(xs)), batch_size, replace=False)
    return xs[indices] , ys[indices]

# 데이터의 범위를 설정합니다.
t_min, t_max = 0, 30
resolution = 0.1

# Training Data
start_train = 0.0
end_train = 25.0  # 12.2 + 0.1 * (n_steps + 1)
train_x_axis = np.arange(start_train, end_train, 0.1)
train_dataset = time_series(train_x_axis)

# Validation Data
start_val = 25.0
end_val = 30.0
val_x_axis = np.arange(start_val, end_val, 0.1)
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

# Model
x = tf.placeholder(tf.float32, [None, n_steps])
x_res = tf.reshape(x, shape=[-1, n_steps, n_inputs])
x_trpose = tf.transpose(x_res , perm=(1, 0, 2))
xs_seq = tf.unstack(x_trpose)

y = tf.placeholder(tf.float32, [None, n_steps])
y_res = tf.reshape(y, shape=[-1, n_steps, n_inputs])

init_wx = tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32, stddev=0.1)
wx = tf.Variable(init_wx)
init_wh = tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32, stddev=0.1)
wh = tf.Variable(init_wh)
init_b = tf.constant(value=0, shape=[n_neurons], dtype=tf.float32, stddev=0.1)
b = tf.Variable(init_b)

# init hidden , have to input 0,
init_hidden = tf.placeholder(shape=[None, n_neurons], dtype=tf.float32)
hidden_state = tf.zeros_like(init_hidden, dtype=tf.float32)

activation = tf.nn.tanh
output_layers = []
logits_list = []

for i, x_seq in enumerate(xs_seq):

    # inputs
    hidden_layer = tf.matmul(hidden_state, wh)
    input_layer = tf.matmul(x_seq, wx)
    output_layer = activation(hidden_layer + input_layer + b)
    output_layers.append(output_layer)
    hidden_state = output_layer

    # logits
    #logits = tf.layers.dense(output_layer, 1, activation=None, use_bias=True)
    #logits_list.append(logits)

stacked_outputs = tf.reshape(output_layers, shape=[-1, n_neurons])
stacked_logits = tf.layers.dense(stacked_outputs, n_outputs)
outputs = tf.reshape(stacked_logits, [-1, n_steps, n_outputs])

loss = tf.reduce_mean(tf.square(outputs - y_res))
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize((loss))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 60
init_hidden_value = np.zeros(shape=[batch_size, n_neurons], dtype=np.float32)
for step in range(3000):
    batch_xs, batch_ys = next_batch(train_xs, train_ys, batch_size)
    _, train_loss = sess.run([train_op, loss],
                         feed_dict={x: batch_xs, y: batch_ys, init_hidden: init_hidden_value})

    print(train_loss)

# TODO
# 데이터 Test Code 작성하기