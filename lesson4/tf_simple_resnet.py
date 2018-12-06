import tensorflow as tf
import numpy as np
import random

def batch_normalization(x, phase_train , scope_name):
    with tf.variable_scope(scope_name):
        n_out=int(x.get_shape()[-1])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
                            lambda: mean_var_with_update(),
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def stem(x, phase_train):
    """

    :param x:
    :param phase_train: Tensor(bool type)
    :return:
    """
    # Convolution Initializer
    he_init = tf.initializers.variance_scaling(scale=2)

    # Layers
    layer = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=2, padding='same', kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='stem1')
    layer = tf.nn.relu(layer)
    layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2)

    return layer


def residual_block(x, out_ch, phase_train):
    """
     Residual block
      .-----------------------------------------------------.
      |                                                     |
      |                                                     |
     __                         __                          |               __
    |  |-->3x3,conv(out_ch)--> |  |-->3x3,conv(out_ch)-->-- + ---------->  |  |
    |__|                       |__|                    element-wise        |__|
    input                                                    add          output


    :param x: 입력되는 Feature map 입니다 , Tensor 형태여야 합니다
    :param out_ch: 최종적으로 출력되는 Channel 의 갯수입니다
    :param phase_train: Tensor(bool type)
    :return: tensor
    """
    # Setting
    kernel_size = (3, 3)
    strides = (1, 1)

    # Initializer
    he_init = tf.initializers.variance_scaling(scale=2)

    # Plain layer 1
    layer = tf.layers.conv2d(x, filters=out_ch, kernel_size=kernel_size, strides=strides, padding='same',
                             kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='stem1')
    layer = tf.nn.relu(layer)

    # Plain layer 2
    layer = tf.layers.conv2d(layer, filters=out_ch, kernel_size=kernel_size, strides=strides, padding='same',
                             kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='stem1')
    layer = tf.nn.relu(layer)

    # Element wise add
    output = tf.add(x, layer)
    return output


def residual_block_projection(x, out_ch, phase_train):
    """
     residual_block_projection
      .------------------- 1x1,conv(out_ch) ----------------.
      |                                                     |
      |                                                     |
     __                         __                          |               __
    |  |-->3x3,conv(out_ch)--> |  |-->3x3,conv(out_ch)-->-- + ---------->  |  |
    |__|                       |__|                    element-wise        |__|
 input(out_ch/2)                                          add             output



    :param x: 입력되는 Feature map 입니다 , Tensor 형태여야 합니다
    :param out_ch: 최종적으로 출력되는 Channel 의 갯수입니다
    :param phase_train: Tensor(bool type)
    :return: tensor
    """
    # Setting
    kernel_size = (3, 3)
    strides = (2, 2)

    # Initializer
    he_init = tf.initializers.variance_scaling(scale=2)

    # Plain Layer 1
    # Reduce Image size : ( h,w --> h/2,w/2 )
    layer = tf.layers.conv2d(x, filters=out_ch, kernel_size=kernel_size, strides=strides, padding='same',
                             kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='bn1')
    layer = tf.nn.relu(layer)

    # Plain Layer 2
    layer = tf.layers.conv2d(layer, filters=out_ch, kernel_size=kernel_size, strides=(1, 1), padding='same',
                             kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='bn2')
    layer = tf.nn.relu(layer)

    # Projection Layer
    # Reduce Image size : ( h,w --> h/2,w/2 )
    projection_layer = tf.layers.conv2d(x, filters=out_ch, kernel_size=(1, 1), strides=strides, padding='same',
                                        kernel_initializer=he_init)
    projection_layer = batch_normalization(projection_layer, phase_train=phase_train, scope_name='bn_projection')
    projection_layer = tf.nn.relu(projection_layer)

    # Element wise add
    output = tf.add(projection_layer, layer)
    return output


def bottlenect_block(x, out_ch, phase_train):
    """
     bottlenect_block
      .---------------------------------------------------------------------------------.
      |                                                                                 |
      |                                                                                 |
     __                           __                          __                        |           __
    |  |-->1x1,conv(out_ch/4)--> |  |-->3x3,conv(out_ch/4)-->|  |-->1x1,conv(out_ch) -- + ------>  |  |
    |__|                         |__|                        |__|                (element-wise)    |__|
 input(out_ch/4)              (out_ch/4)                   input(out_ch/4)                        output(out_ch)



    :param x: 입력되는 Feature map 입니다 , Tensor 형태여야 합니다
    :param out_ch: 최종적으로 출력되는 Channel 의 갯수입니다
    :param phase_train: Tensor(bool type)
    :return: tensor
    """
    # Setting
    kernel_size = (3, 3)
    strides = (1, 1)
    activation = tf.nn.relu
    he_init = tf.initializers.variance_scaling(scale=2)

    # Plain Layer 1
    layer = tf.layers.conv2d(x, filters=int(out_ch // 4), kernel_size=(1, 1), strides=strides, padding='same',
                             activation=activation, kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='stem1')
    layer = tf.nn.relu(layer)

    # Plain Layer 2
    layer = tf.layers.conv2d(layer, filters=int(out_ch // 4), kernel_size=kernel_size, strides=strides, padding='same',
                             activation=activation, kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='stem1')
    layer = tf.nn.relu(layer)

    # Plain Layer 3
    layer = tf.layers.conv2d(layer, filters=out_ch, kernel_size=(1, 1), strides=strides, padding='same',
                             activation=activation, kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='stem1')
    layer = tf.nn.relu(layer)

    # element wise add
    output = tf.add(x, layer)
    return output


def bottlenect_block_projection(x, out_ch, phase_train):
    """
     bottlenect_block_projection
      .--------------------------------------1x1,conv,(out_ch)--------------------------.
      |                                                                                 |
      |                                                                                 |
     __                           __                          __                        |           __
    |  |-->1x1,conv(out_ch/4)--> |  |-->3x3,conv(out_ch/4)-->|  |-->1x1,conv(out_ch) -- + ------>  |  |
    |__|                         |__|                        |__|                (element-wise)    |__|
 input(out_ch/4)              (out_ch/4)                   input(out_ch/4)                        output(out_ch)



    :param x: 입력되는 Feature map 입니다 , Tensor 형태여야 합니다
    :param out_ch: 최종적으로 출력되는 Channel 의 갯수입니다
    :param phase_train: Tensor(bool type)
    :return: tensor
    """

    kernel_size = (3, 3)
    strides = (2, 2)
    activation = tf.nn.relu
    he_init = tf.initializers.variance_scaling(scale=2)

    # Plain Layers
    layer = tf.layers.conv2d(x, filters=int(out_ch // 4), kernel_size=(1, 1), strides=strides, padding='same',
                             activation=activation, kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='stem1')
    layer = tf.nn.relu(layer)

    layer = tf.layers.conv2d(layer, filters=int(out_ch // 4), kernel_size=kernel_size, strides=(1, 1), padding='same',
                             activation=activation, kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='stem1')
    layer = tf.nn.relu(layer)

    layer = tf.layers.conv2d(layer, filters=int(out_ch), kernel_size=(1, 1), strides=(1, 1), padding='same',
                             activation=activation, kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='stem1')
    layer = tf.nn.relu(layer)

    # Projection layer
    projection_input = tf.layers.conv2d(x, filters=out_ch, kernel_size=(1, 1), strides=strides, padding='same',
                                        activation=activation, kernel_initializer=he_init)
    layer = batch_normalization(layer, phase_train=phase_train, scope_name='stem1')
    layer = tf.nn.relu(layer)

    # element wise add
    output = tf.add(projection_input, layer)
    return output


def resnet_18(input_shape, n_classes):

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool)

    # Stem Block
    layer = stem(x, phase_train=phase_train)

    # Block 1
    layer = residual_block(layer, out_ch=64, phase_train=phase_train)
    layer = residual_block(layer, out_ch=64, phase_train=phase_train)

    # Block 2
    layer = residual_block_projection(layer, out_ch=128, phase_train=phase_train)
    layer = residual_block(layer, out_ch=128, phase_train=phase_train)

    # Block 3
    layer = residual_block_projection(layer, out_ch=256, phase_train=phase_train)
    layer = residual_block(layer, out_ch=256, phase_train=phase_train)

    # Block 4
    layer = residual_block_projection(layer, out_ch=512, phase_train=phase_train)
    layer = residual_block(layer, out_ch=512, phase_train=phase_train)

    # Change node name
    top_conv = tf.identity(layer, 'top_conv')

    # Global Average Pooling
    h,w=top_conv.get_shape()[1:3]
    gap_layer = tf.layers.average_pooling2d(top_conv, pool_size=(h, w), strides=(1, 1))

    # Flatten layer
    flat_layer = tf.contrib.layers.flatten(gap_layer)

    # logits
    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(flat_layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

    # Mean cost values
    costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    cost_op = tf.reduce_mean(costs_op)

    # Accuracy
    y_cls = tf.argmax(y, axis=1)
    correct = tf.nn.in_top_k(logits, y_cls, 1)
    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    # return ops
    ops = {'x': x, 'y': y, 'phase_train': phase_train, 'cost_op': cost_op, 'acc_op': acc_op}
    return ops


def resnet_34(input_shape, n_classes):


    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool)

    # Stem Block
    layer = stem(x, phase_train=phase_train)

    # Block 1
    for i in range(3):
        layer = residual_block(layer, out_ch=64, phase_train=phase_train)

    # Block 2
    layer = residual_block_projection(layer, out_ch=128, phase_train=phase_train)
    for i in range(4):
        layer = residual_block(layer, out_ch=128, phase_train=phase_train)

    # Block 3
    layer = residual_block_projection(layer, out_ch=256, phase_train=phase_train)
    for i in range(6):
        layer = residual_block(layer, out_ch=256, phase_train=phase_train)

    # Block 4
    layer = residual_block_projection(layer, out_ch=512, phase_train=phase_train)
    for i in range(3):
        layer = residual_block(layer, out_ch=512, phase_train=phase_train)

    # Change node name
    top_conv = tf.identity(layer, 'top_conv')

    # Global Average Pooling
    h,w=top_conv.get_shape()[1:3]
    gap_layer = tf.layers.average_pooling2d(top_conv, pool_size=(h, w), strides=(1, 1))

    # Flatten layer
    flat_layer = tf.contrib.layers.flatten(gap_layer)

    # logits
    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(flat_layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

    # Mean cost values
    costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    cost_op = tf.reduce_mean(costs_op)

    # Accuracy
    y_cls = tf.argmax(y, axis=1)
    correct = tf.nn.in_top_k(logits, y_cls, 1)
    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    # return ops
    ops = {'x': x, 'y': y, 'phase_train': phase_train, 'cost_op': cost_op, 'acc_op': acc_op}
    return ops


def resnet_50(input_shape, n_classes):

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool)

    # Stem Block
    layer = stem(x, phase_train=phase_train)

    # Block 1
    layer = bottlenect_block_projection(layer, out_ch=256, phase_train=phase_train)
    for i in range(2):
        layer = bottlenect_block(layer, out_ch=256, phase_train=phase_train)

    # Block 2
    layer = bottlenect_block_projection(layer, out_ch=512, phase_train=phase_train)
    for i in range(3):
        layer = bottlenect_block(layer, out_ch=512, phase_train=phase_train)

    # Block 3
    layer = bottlenect_block_projection(layer, out_ch=1024, phase_train=phase_train)
    for i in range(5):
        layer = bottlenect_block(layer, out_ch=1024, phase_train=phase_train)

    # Block 4
    layer = bottlenect_block_projection(layer, out_ch=2048, phase_train=phase_train)
    for i in range(5):
        layer = bottlenect_block(layer, out_ch=2048, phase_train=phase_train)

    # Change node name
    top_conv = tf.identity(layer, 'top_conv')

    # Global Average Pooling
    h,w=top_conv.get_shape()[1:3]
    gap_layer = tf.layers.average_pooling2d(top_conv, pool_size=(h, w), strides=(1, 1))

    # Flatten layer
    flat_layer = tf.contrib.layers.flatten(gap_layer)

    # logits
    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(flat_layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

    # Mean cost values
    costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    cost_op = tf.reduce_mean(costs_op)

    # Accuracy
    y_cls = tf.argmax(y, axis=1)
    correct = tf.nn.in_top_k(logits, y_cls, 1)
    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    # return ops
    ops = {'x': x, 'y': y, 'phase_train': phase_train, 'cost_op': cost_op, 'acc_op': acc_op}
    return ops


def resnet_101(input_shape, n_classes):

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool)

    # Stem Block
    layer = stem(x, phase_train=phase_train)

    # Block 1
    layer = bottlenect_block_projection(layer, out_ch=256, phase_train=phase_train)
    for i in range(2):
        layer = bottlenect_block(layer, out_ch=256, phase_train=phase_train)

    # Block 2
    layer = bottlenect_block_projection(layer, out_ch=512, phase_train=phase_train)
    for i in range(3):
        layer = bottlenect_block(layer, out_ch=512, phase_train=phase_train)

    # Block 3
    layer = bottlenect_block_projection(layer, out_ch=1024, phase_train=phase_train)
    for i in range(22):
        layer = bottlenect_block(layer, out_ch=1024, phase_train=phase_train)

    # Block 4
    layer = bottlenect_block_projection(layer, out_ch=2048, phase_train=phase_train)
    for i in range(2):
        layer = bottlenect_block(layer, out_ch=2048, phase_train=phase_train)

    # Change node name
    top_conv = tf.identity(layer, 'top_conv')

    # Global Average Pooling
    h,w=top_conv.get_shape()[1:3]
    gap_layer = tf.layers.average_pooling2d(top_conv, pool_size=(h, w), strides=(1, 1))

    # Flatten layer
    flat_layer = tf.contrib.layers.flatten(gap_layer)

    # logits
    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(flat_layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

    # Mean cost values
    costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    cost_op = tf.reduce_mean(costs_op)

    # Accuracy
    y_cls = tf.argmax(y, axis=1)
    correct = tf.nn.in_top_k(logits, y_cls, 1)
    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    # return ops
    ops = {'x': x, 'y': y, 'phase_train': phase_train, 'cost_op': cost_op, 'acc_op': acc_op}
    return ops


def resnet_152(input_shape, n_classes):

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool)

    # Stem Block
    layer = stem(x, phase_train=phase_train)

    # Block 1
    layer = bottlenect_block_projection(layer, out_ch=256, phase_train=phase_train)
    for i in range(2):
        layer = bottlenect_block(layer, out_ch=256, phase_train=phase_train)

    # Block 2
    layer = bottlenect_block_projection(layer, out_ch=512, phase_train=phase_train)
    for i in range(7):
        layer = bottlenect_block(layer, out_ch=512, phase_train=phase_train)

    # Block 3
    layer = bottlenect_block_projection(layer, out_ch=1024, phase_train=phase_train)
    for i in range(35):
        layer = bottlenect_block(layer, out_ch=1024, phase_train=phase_train)

    # Block 4
    layer = bottlenect_block_projection(layer, out_ch=2048, phase_train=phase_train)
    for i in range(2):
        layer = bottlenect_block(layer, out_ch=2048, phase_train=phase_train)
    # Change node name
    top_conv = tf.identity(layer, 'top_conv')

    # Global Average Pooling
    h,w=top_conv.get_shape()[1:3]
    gap_layer = tf.layers.average_pooling2d(top_conv, pool_size=(h, w), strides=(1, 1))

    # Flatten layer
    flat_layer = tf.contrib.layers.flatten(gap_layer)

    # logits
    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(flat_layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

    # Mean cost values
    costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    cost_op = tf.reduce_mean(costs_op)

    # Accuracy
    y_cls = tf.argmax(y, axis=1)
    correct = tf.nn.in_top_k(logits, y_cls, 1)
    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    # return ops
    ops = {'x': x, 'y': y, 'phase_train': phase_train, 'cost_op': cost_op, 'acc_op': acc_op}
    return ops


def compile(optimizer_name, ops, learning_rate):
    cost_op = ops['cost_op']
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'sgd':
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)

    elif optimizer_name == 'momentum':
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(cost_op)

    elif optimizer_name == 'rmsprop':
        train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_op)

    elif optimizer_name == 'adadelta':
        train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost_op)

    elif optimizer_name == 'adagrad':
        train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost_op)

    elif optimizer_name == 'adam':
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

    # elif optimizer_name == 'adagradda':
    #    train_op = tf.train.AdagradDAOptimizer(learning_rate).minimize(cost_op)

    else:
        raise ValueError

    # add train_op to ops
    ops['train_op'] = train_op

    return ops


def create_session():

    """config Option
     allow_soft_placement :  if cannot put a node in a gpu , put node to in a cpu
     log_device_placement :  show where each node is assigned
     config.gpu_options.allow_growth : 처음부터 메모리를 점유하지 말고 필요한 메모리를 점차 증가 시킵니다
    """

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    return sess


def next_batch(imgs, labs, batch_size):
    # random shuffle list
    indices = random.sample(range(np.shape(imgs)[0]), batch_size)

    imgs = np.asarray(imgs)
    batch_xs = imgs[indices]
    batch_ys = labs[indices]
    return batch_xs, batch_ys

def training(sess, n_step, train_images, train_labels, batch_size, ops):
    """
    Usage :
    >>> training(sess, n_step, batch_xs, batch_ys, ops)
    :param sess: tf.Session
    :param n_step: int | E.g)
    :param train_images: Numpy | E.g)
    :param train_labels: Numpy | E.g)
    :param ops: tensor operations | E.g)
    :return: cost values
    """

    cost_values = []
    for i in range(n_step):

        # Extract batch images , labels
        batch_xs, batch_ys = next_batch(train_images, train_labels, batch_size)
        # Training
        fetches = [ops['train_op'], ops['cost_op']]
        feed_dict = {ops['x']: batch_xs, ops['y']: batch_ys, ops['phase_train']: True}
        _, cost = sess.run(fetches, feed_dict)
        cost_values.append(cost)

    return cost_values


def eval(sess, batch_xs, batch_ys, ops):
    """
    Usage :
    >>> eval(sess, batch_xs, batch_ys, ops)
    :param sess: tf.Session
    :param batch_xs: xs | E.g)
    :param batch_ys: ys | E.g)
    :param ops: tensor operations | E.g)
    :return: cost values
    """

    fetches = [ops['acc_op'], ops['cost_op']]
    feed_dict = {ops['x']: batch_xs, ops['y']: batch_ys, ops['phase_train']: False}

    return sess.run(fetches, feed_dict)
