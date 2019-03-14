import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
OutputProjectionWrapper Vs BasicRNN Cell

Projection 이란 투영 이란 개념으로 Resnet 에서도 사용되는 개념입니다 
자신을 투영해 확대하거나 축소하는 개념을 가지고 있습니다 
Projecter 의 개념을 생각해 보세요. 


OutputProjectionWrapper 
                        fc
               state   layer
   --------------------------

     ___       ___      ___ 
t1: |___| ->  |___| -> |___|(outputs[0]) 
    <----------------------->
             wrapper 
     ___       ___      ___ 
t2: |___| ->  |___| -> |___|(outputs[1])
    <----------------------->
             wrapper 

            ....
            if timestep = 9

     ___       ___      ___ 
t9: |___| ->  |___| -> |___|(output[9])
              states
    <----------------------->
             wrapper 



cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
warpped_cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size = n_outputs)
outputs, states = tf.nn.dynamic_rnn(warpped_cell, x, dtype=tf.float32)

"""

t_min, t_max = 0, 30
resolution = 0.1
n_steps = 21
n_inputs = 1
n_neurons = 100
n_outputs = 1

# Training Data
# Training , Validation 할 범위를 설정합니다
# time step 이 21 인경우에 12.2 , 14.2 범위는 train 데이터가 하나밖에 나오지 않아 overfitting 을 유도합니다
# 그래서 모델이 잘 학습되나 학습되지 않나 확인하는 목적입니다

train_x_axis = np.arange(12.2, 14.2, 0.1)
val_x_axis = np.hstack([np.arange(0, 12.1, 0.1), np.arange(14.3, 30, 0.1)])


def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t * 5)

# 지정했던 범위에 time_series 을 적용해 변환합니다
train_dataset = time_series(train_x_axis)
val_dataset = time_series(val_x_axis)


# dataset을 생성합니다
# 이 함수는 datum 을 받아 timestep 에 맞게 쪼개는 함수입니다
# 가령 x = [0.0 ,0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 이고 timestep 이 5 이라면
# x = [0.0 ,0.1, 0.2, 0.3, 0.4] , y = 0.5
# x = [0.1, 0.2, 0.3, 0.4, 0.5] , y = 0.6
# x = [0.2, 0.3, 0.4, 0.5, 0.6] , y = 0.7
# x = [0.3, 0.4, 0.5, 0.6, 0.7] , y = 0.8
# x = [0.4, 0.5, 0.6, 0.7, 0.8] , y = 0.9
# x = [0.5, 0.6, 0.7, 0.8, 0.9] , y = 1.0
# 이렇게 쪼개집니다

def generate_predict(datum, timestep):
    xs = []
    ys = []
    for i in range(len(datum) - timestep):
        """
        please first read this 
        if x = [ 0 ,0.1, 0.2, 0.3, 0.4 , 0.5] and timestpe is 5 
          datum[0: 0 + timestep]은 x = [0 ,0.1, 0.2, 0.3, 0.4] 
          datum[i + timestep] 은 y = [0.5] 이 된다 
          이렇게 데이터셋을 구축해야 모델이 다음 timestep 의 value 을 예측 할수 있게 된다 
        """
        xs.append(datum[i: i + timestep])  # if time step = 5 [0 ,0.1, 0.2, 0.3, 0.4]
        ys.append(datum[i + timestep])  # if time step = 5[0.5]
    return np.asarray(xs), np.asarray(ys)

train_xs, train_ys = generate_predict(train_dataset, n_steps)
val_xs, val_ys = generate_predict(val_dataset, n_steps)


# xs, ys 데이터에서 임의로 값을 가져옵니다, 중복을 허용합니다(replace=True)
def next_batch(xs, ys, batch_size):
    indices = np.random.choice(len(xs), batch_size, replace=True)
    return xs[indices], ys[indices]


x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None])

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
warpped_cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=n_outputs)

outputs, states = tf.nn.dynamic_rnn(warpped_cell, x, dtype=tf.float32)

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


# 최종적으로 평가를합니다 validation 의 모든 데이터셋을 하나하나씩 꺼내 평가하고
# 모델을 통과하고 나온 예측값을  predicts 에 append 합니다

# Eval
predicts = []
for xs in val_xs:
    xs = xs.reshape(1, n_steps, n_inputs)
    outputs_, states_ = sess.run([outputs, states], feed_dict={x: xs})
    predicts.append(np.squeeze(outputs_)[-1])

# visualization Validation Dataset
# x 축을 값을 가져옵니다 , 그리고 정답과 예측값을 동시에 비교합니다
x_axis = val_x_axis[(n_steps - 1): (n_steps - 1) + len(val_ys)]
plt.scatter(x_axis, val_ys, c='r', label='true')
plt.scatter(x_axis, predicts, c='b', label='predict')
plt.legend()
plt.show()
