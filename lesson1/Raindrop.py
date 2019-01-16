import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

N = 500
u_init = np.zeros([500,500] , dtype=np.float32)
ut_init = np.zeros([500,500] , dtype=np.float32)

for n in range(40):
    a, b = np.random.randint(0, N, 2)
    u_init[a,b] = np.random.uniform()

def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]


def laplace(x):
    """Compute the 2D laplacian of an array"""
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                             [1.0, -6., 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)


# Parameters:
# eps -- time resolution
# damping -- wave damping

eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# Discretized PDE update rulesw
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# Operation to update the state
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

for i in range(8000):
    sess.run(step, {eps: 0.03, damping: 0.04})
    if i % 500 == 0 :
        result = sess.run(U)
        result = (result + 0.1) / float(0.2) * 255
        #result = np.uint8(np.clip(result, 0, 255))

        #norm_result = (result - np.min(result)) / (np.max(result) - np.min(result))

        plt.imshow(result, cmap='Greys')
        plt.show()



