import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ys = [23, 26, 30, 34, 43, 48, 52, 57 ,58]
xs = [6.51, 7.62, 8.56, 10.63, 11.90, 12.98, 14.21, 14.40, 15.18]

plt.ylabel('Advertising fee')
plt.xlabel('Sales')
plt.scatter(xs, ys)
plt.show()
plt.close()

# AX + B
# A, B 변수로
# X placeholder 로 만들기

# X =23 일때
X = tf.placeholder(shape=[], name='X', dtype=tf.float32)
Y = tf.placeholder(shape=[], name='Y', dtype=tf.float32)
A = tf.placeholder(shape=[], name='A', dtype=tf.float32)
B = tf.placeholder(shape=[], name='B', dtype=tf.float32)

multiply = tf.multiply(A, X, name='multiply')
pred = tf.add(multiply, B)

loss_sub = tf.subtract(pred, Y)
loss = tf.square(loss_sub)

# X가 651 일때 A -100 ~ 100 , B =1 , Loss?

sess = tf.Session()
losses=[]
As = range(-300, 300, 1)
for a in As:
    a = a * 0.1
    loss_ = sess.run(loss, feed_dict={X: 6.51, Y: 23, A: a, B: 1})
    losses.append(loss_)

plt.scatter(As, losses, s=0.5)
plt.show()
plt.close()

# X가 651 일때 A = 1 , B =-30~ 30, Loss?
losses=[]
Bs = range(-300, 300, 1)
for b in Bs:
    b = b * 0.1
    loss_ = sess.run(loss, feed_dict={X: 6.51, Y: 23, A: 1, B: b})
    losses.append(loss_)
plt.scatter(Bs, losses, s=0.5)
plt.show()
plt.close()







