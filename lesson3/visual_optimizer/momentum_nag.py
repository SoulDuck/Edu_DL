import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


# Gradient Descent
def gradient_descent(start_x, start_y, lr, func):
    with tf.name_scope('GD'):
        x_ = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y_ = tf.Variable(initial_value=tf.constant(start_y), name='y')

        # Get Gradient x,y with tf_f
        z = func(x_, y_)
        z = tf.identity(z, name='z')
        tf.train.GradientDescentOptimizer(lr).minimize(z, name='train_op')


def momentum(start_x, start_y, lr, func, momentum=0.9):
    with tf.name_scope('Momentum'):
        x_ = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y_ = tf.Variable(initial_value=tf.constant(start_y), name='y')

        # Get Gradient x,y with tf_f
        z = func(x_, y_)
        z = tf.identity(z, name='z')
        tf.train.MomentumOptimizer(lr, momentum=momentum).minimize(z, name='train_op')


def momentum_nag(start_x, start_y, lr, func, momentum=0.9):
    with tf.name_scope('NAG'):
        x_ = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y_ = tf.Variable(initial_value=tf.constant(start_y), name='y')

        # Get Gradient x,y with func
        z = func(x_, y_)
        z = tf.identity(z, name='z')
        tf.train.MomentumOptimizer(lr, momentum=momentum, use_nesterov=True).minimize(z, name='train_op')


def adagard(start_x, start_y, lr, func):

    with tf.name_scope('Adagrad'):
        x_ = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y_ = tf.Variable(initial_value=tf.constant(start_y), name='y')

        # Get Gradient x,y with func
        z = func(x_, y_)
        z = tf.identity(z, name='z')
        tf.train.AdagradOptimizer(lr).minimize(z, name='train_op')


def adagard_da(start_x, start_y, lr, func, global_step):

    with tf.name_scope('AdagradDA'):
        x_ = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y_ = tf.Variable(initial_value=tf.constant(start_y), name='y')

        # Get Gradient x,y with func
        z = func(x_, y_)
        z = tf.identity(z, name='z')
        tf.train.AdagradDAOptimizer(lr, global_step=global_step).minimize(z, name='train_op')


def adadelta(start_x, start_y, lr, func):
    with tf.name_scope('Adadelta'):
        x_ = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y_ = tf.Variable(initial_value=tf.constant(start_y), name='y')

        # Get Gradient x,y with func
        z = func(x_, y_)
        z = tf.identity(z, name='z')
        tf.train.AdadeltaOptimizer(lr).minimize(z, name='train_op')


def rms_prop(start_x, start_y, lr, func):
    with tf.name_scope('RMSProp'):
        x_ = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y_ = tf.Variable(initial_value=tf.constant(start_y), name='y')

        # Get Gradient x,y with func
        z = func(x_, y_)
        z = tf.identity(z, name='z')
        tf.train.RMSPropOptimizer(lr).minimize(z, name='train_op')


def adam(start_x, start_y, lr, func):
    with tf.name_scope('Adam'):
        x_ = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y_ = tf.Variable(initial_value=tf.constant(start_y), name='y')

        # Get Gradient x,y with func
        z = func(x_, y_)
        z = tf.identity(z, name='z')
        tf.train.AdamOptimizer(lr).minimize(z, name='train_op')


def ftrl(start_x, start_y, lr, func):
    with tf.name_scope('Ftrl'):
        x_ = tf.Variable(initial_value=tf.constant(start_x), name='x')
        y_ = tf.Variable(initial_value=tf.constant(start_y), name='y')

        # Get Gradient x,y with func
        z = func(x_, y_)
        z = tf.identity(z, name='z')
        tf.train.FtrlOptimizer(lr).minimize(z, name='train_op')


def trace(session, max_iter, optimizer_name):

    graph = tf.get_default_graph()
    x_tf = graph.get_tensor_by_name('{}/x:0'.format(optimizer_name))
    y_tf = graph.get_tensor_by_name('{}/y:0'.format(optimizer_name))
    z_tf = graph.get_tensor_by_name('{}/z:0'.format(optimizer_name))
    step = graph.get_operation_by_name('{}/train_op'.format(optimizer_name))

    xs_, ys_, zs_ = [], [], []
    for i in range(max_iter):
        _, x_, y_, z_ = session.run([step, x_tf, y_tf, z_tf])
        xs_.append(x_)
        ys_.append(y_)
        zs_.append(z_)
    return xs_, ys_, zs_


def beale(x, y):
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


def generate_beale():
    xmin, xmax, xstep = -4.5, 4.5, .2
    ymin, ymax, ystep = -4.5, 4.5, .2

    xs_ = np.arange(xmin, xmax, xstep)
    ys_ = np.arange(ymin, ymax, ystep)

    x, y = np.meshgrid(xs_, ys_)
    z = beale(x, y)

    minima = np.array([3., .5])
    minima_ = minima.reshape(-1, 1)
    z_minima = beale(*minima)
    # z_minima = minima.reshape(-1, 1)

    plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d', elev=50, azim=-50)

    ax.plot(*minima_, z_minima, 'r*', markersize=10)
    ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1,
                    edgecolor='None', alpha=0.3, cmap=plt.cm.jet)

    ax.view_init(30, 10)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    return ax


def generate_himmelblau():
    pass;


def tf_beale(x, y):
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


# Training
start_x = -3.
start_y = -3.
mmt = 0.9
start_z = beale(start_x, start_y)
momentum(start_x, start_y, 0.00005, tf_beale, mmt)
momentum_nag(start_x, start_y, 0.00005, tf_beale, mmt)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
mm_xs, mm_ys, mmz_zs = trace(sess, 20000, 'Momentum')
nag_xs, nag_ys, nag_zs = trace(sess, 20000, 'NAG')

# Visualization
ax = generate_beale()

# add initial point [start_x] ,[start_y], [start_z]
ax.plot([start_x]+mm_xs, [start_y]+mm_ys, [start_z]+mmz_zs, label='momentum(beta={})'.format(mmt), color='b')
ax.plot([start_x]+nag_xs, [start_y]+nag_ys, [start_z]+nag_zs, label='nag(beta={})'.format(mmt), color='g')
ax.plot([start_x],[start_y],[start_z], 'b.-')
ax.legend()

xs = zip([mm_xs, nag_xs])
for x in xs:
    print(x)

plt.show()
