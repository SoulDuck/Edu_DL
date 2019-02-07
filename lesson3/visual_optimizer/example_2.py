import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


def tf_f(x, y):
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


# Gradient Descent
def gradient_descent(start_x, start_y, lr):
    with tf.name_scope('GD'):

        x = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y = tf.Variable(initial_value=tf.constant(start_y), name='y')

        # Get Gradient x,y with tf_f
        z = tf_f(x, y)
        z = tf.identity(z, name='z')
        grad_x = tf.gradients(z, x, name='dx')
        grad_y = tf.gradients(z, y, name='dy')

        # GD
        lr = tf.constant(value = lr, name='lr')

        # X hat , Y hat
        x_hat = x - grad_x*lr
        x_hat = tf.identity(x_hat, name='x_hat')
        y_hat = y - grad_y*lr
        y_hat = tf.identity(y_hat, name='y_hat')

        step = tf.group(
            tf.assign(x, tf.reshape(x_hat, [])),
            tf.assign(y, tf.reshape(y_hat, [])),
            name='step'
        )

    return step


def momentum(start_x, start_y, lr):
    with tf.name_scope('Momentum'):
        x = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y = tf.Variable(initial_value=tf.constant(start_y), name='y')

        vx = tf.Variable(initial_value=tf.constant(0.0), name='vx')
        vy = tf.Variable(initial_value=tf.constant(0.0), name='vy')

        beta = tf.constant(0.9, name='beta')

        # Get Gradient x,y with tf_f
        z = tf_f(x, y)
        z = tf.identity(z, name='z')
        grad_x = tf.gradients(z, x, name='dx')
        grad_y = tf.gradients(z, y, name='dy')

        # GD
        lr = tf.constant(value=lr, name='lr')

        # X hat , Y hat
        vx_hat = tf.reshape(tensor=beta * vx - lr * grad_x, shape=[])
        vx = tf.assign(ref=vx, value=vx_hat)
        x_hat = x + vx

        vy_hat = tf.reshape(tensor=beta * vy - lr * grad_y, shape=[])
        vy = tf.assign(ref=vy, value=vy_hat)
        y_hat = y + vy

        # Change tensor name
        x_hat = tf.identity(x_hat, name='x_hat')
        y_hat = tf.identity(y_hat, name='y_hat')

        step = tf.group(
            tf.assign(x, tf.reshape(x_hat, [])),
            tf.assign(y, tf.reshape(y_hat, [])),
            name='step'
        )
    return step


# NAG = Nesterov Accelated Gradient
def menmentum_with_NAG(start_x, start_y, lr):
    with tf.name_scope('NAG'):
        x = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y = tf.Variable(initial_value=tf.constant(start_y), name='y')
        v = tf.Variable(initial_value=tf.constant(0.0), name='v')

        beta = tf.constant(0.9, name='beta')


def trace(sess, max_iter, optimizer_name):

    graph = tf.get_default_graph()
    grad_x = graph.get_tensor_by_name('{}/dx/AddN:0'.format(optimizer_name))
    grad_y = graph.get_tensor_by_name('{}/dy/AddN:0'.format(optimizer_name))
    x = graph.get_tensor_by_name('{}/x:0'.format(optimizer_name))
    y = graph.get_tensor_by_name('{}/y:0'.format(optimizer_name))
    z = graph.get_tensor_by_name('{}/z:0'.format(optimizer_name))
    step = graph.get_operation_by_name('{}/step'.format(optimizer_name))

    xs_, ys_, zs_, dx_, dy_ = [], [], [], [], []
    for i in range(max_iter):
        _, dx, dy, x_, y_, z_ = sess.run([step, grad_x, grad_y, x, y, z])
        xs_.append(x_)
        ys_.append(y_)
        zs_.append(z_)
        dx_.append(dx)
        dy_.append(dy)
    return xs_, ys_, zs_, dx_, dy_


# Visualization
def generate_backgroud():
    f = lambda x, y: (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2

    xmin, xmax, xstep = -4.5, 4.5, .2
    ymin, ymax, ystep = -4.5, 4.5, .2

    xs = np.arange(xmin, xmax, xstep)
    ys = np.arange(ymin, ymax, ystep)

    x, y = np.meshgrid(xs, ys)
    z = f(x, y)

    minima = np.array([3., .5])
    minima_ = minima.reshape(-1, 1)
    z_minima = f(*minima)
    # z_minima = minima.reshape(-1, 1)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d', elev=50, azim=-50)

    ax.plot(*minima_, z_minima, 'r*', markersize=10)
    ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1,
                    edgecolor='None', alpha=0.5, cmap=plt.cm.jet)

    ax.view_init(30, 10)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    return ax


def add_quiver(ax, i, xs, ys, zs):
    xs, ys, zs = map(np.asarray, [xs, ys, zs])
    ax.quiver(xs[1:], ys[i], zs[i], xs[:-1] - xs[1:], ys[:-1] - ys[1:], zs[:-1] - zs[1:])


momentum(-3., -3., 0.000001)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
xs_, ys_, zs_, dx_, dy_ = trace(sess, 80000, 'Momentum')

# Visualization
ax = generate_backgroud()

ax.plot(xs_, ys_, zs_)

plt.show()





"""


f = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

xmin, xmax, xstep = -4.5, 4.5, .2
ymin, ymax, ystep = -4.5, 4.5, .2

xs = np.arange(xmin, xmax, xstep)
ys = np.arange(ymin, ymax, ystep)

x, y = np.meshgrid(xs, ys)
z = f(x, y)

minima = np.array([3., .5])
minima_ = minima.reshape(-1, 1)
z_minima = f(*minima)
#z_minima = minima.reshape(-1, 1)

fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d', elev=50, azim=-50)

ax.plot(*minima_, z_minima, 'r*', markersize=10)
ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1,
                edgecolor='c', alpha=0.5, cmap=plt.cm.jet, linewidth=0.5)

ax.view_init(30, 10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

plt.show()
plt.close()

# get gradient each meshgrid point


"""