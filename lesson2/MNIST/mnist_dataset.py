import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)



print('Train ')
print(mnist.train.images.shape)
print(np.histogram(mnist.train.labels)[0])
print(mnist.train.images.mean())
print(mnist.train.images.std())

print('validation ')
print(mnist.validation.images.shape)
print(np.histogram(mnist.validation.labels)[0])
print(mnist.validation.images.mean())
print(mnist.validation.images.std())










