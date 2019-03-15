import tensorflow as tf

x1_data = [1, 0, 3, 0, 5]
x2_data = [0, 2, 0, 4, 0]
y_data  = [1, 2, 3, 4, 5]

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

b  = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# feature 갯수만큼 곱하는 이 부분을 제외하면 one-variable과 다른 곳이 없다
hypothesis = W1*x1_data + W2*x2_data + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):

    sess.run(train)

    if step%20 == 0:
        print(step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b))

sess.close()

