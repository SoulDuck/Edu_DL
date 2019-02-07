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
start_z = beale(start_x, start_y)

ada_lr = 0.01
rmsp_lr = 0.01
mmt_lr = 0.00005

mmt=0.7
momentum_nag(start_x, start_y, mmt_lr , tf_beale, mmt)
rms_prop(start_x, start_y, rmsp_lr, tf_beale)
adam(start_x, start_y, ada_lr, tf_beale)

# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
mmt_xs, mmt_ys, mmt_zs = trace(sess, 50000, 'NAG')
rmsp_xs, rmsp_ys, rmsp_zs = trace(sess, 50000, 'RMSProp')
adam_xs, adam_ys, adam_zs = trace(sess, 50000, 'Adam')

# Visualization
ax = generate_beale()

# add initial point [start_x] ,[start_y], [start_z]
ax.plot([start_x]+mmt_xs, [start_y]+mmt_ys, [start_z]+mmt_zs,
        label='Momentum with NAG(beta={} ,lr={})'.format(mmt, ada_lr), color='g')
ax.plot([start_x]+rmsp_xs, [start_y]+rmsp_ys, [start_z]+rmsp_zs, label='RMSProp(lr={})'.format(rmsp_lr), color='b')
ax.plot([start_x]+adam_xs, [start_y]+adam_ys, [start_z]+adam_zs, label='Adam(lr={})'.format(rmsp_lr), color='y')
ax.plot([start_x],[start_y],[start_z], 'b.-')
ax.legend()

xs = zip([mmt_xs, rmsp_xs, adam_xs])
for x in xs:
    print(x)

plt.show()
