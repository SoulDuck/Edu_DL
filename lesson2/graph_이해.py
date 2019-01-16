import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ys = [23, 26, 30, 34, 43, 48, 52, 57 ,58]
xs = [651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518]

plt.ylabel('Advertising fee')
plt.xlabel('Sales')
plt.scatter(xs, ys)
plt.show()
plt.close()

# AX + B
# A, B 변수로
# X placeholder 로 만들기

X = tf.placeholder(shape=[] , name='X' ,dtype='float32')
Y = tf.placeholder(shape=[] , name='Y' ,dtype='float32')

random_init_0 = tf.random_normal(shape=[], mean=0, stddev=0.1)
A = tf.Variable(random_init_0 , name='A')

random_init_1 = tf.random_normal(shape=[], mean=0, stddev=0.1)
B = tf.Variable(random_init_1, name='B')

multiply = tf.multiply(A, X, name='multiply')
pred = tf.add(multiply, B)

loss_sub = tf.subtract(pred, Y)
loss = tf.square(loss_sub)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
ys_hat = []
lossses = []
xs_ys = zip(xs,ys)
for x,y in xs_ys:
    y_hat, loss_ = sess.run([pred, loss], feed_dict={X: x, Y: y})
    ys_hat.append(y_hat)
    lossses.append(loss_)

"""
loss = pred - y
loss = loss * loss 
"""
plt.ylabel('Advertising fee')
plt.xlabel('Sales')
plt.scatter(xs, ys)

plt.ylabel('Prediction Advertising fee')
plt.xlabel('Sales')
plt.scatter(xs, ys_hat)
plt.show()
plt.close()


plt.ylabel('loss')
plt.xlabel('Sales')
plt.scatter(xs, lossses)
plt.show()







