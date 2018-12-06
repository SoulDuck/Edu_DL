import tensorflow as tf
import os

def vgg_11(input_shape, n_classes):
    """

    :param input_shape: list or tuple , MUST be 4 elements | E.g) [None , 32, 32, 3]
    :param n_classes: int | 110
    :return:
    """
    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')

    activation = tf.nn.relu
    layer = tf.layers.conv2d(x, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # FC Layer
    # flat_layer = tf.layers.Flatten(layer)
    flat_layer = tf.contrib.layers.flatten(layer)
    xavier_init = tf.initializers.variance_scaling(scale=1)

    with tf.variable_scope('fc1'):
        fc1 = tf.layers.dense(flat_layer, 4096, activation=activation, use_bias=True, kernel_initializer=xavier_init)
        fc1 = tf.cond(phase_train, lambda: tf.nn.dropout(fc1, 0.5), lambda: fc1)
    with tf.variable_scope('fc2'):
        fc2 = tf.layers.dense(flat_layer, 4096, activation=activation, use_bias=True, kernel_initializer=xavier_init)
        fc2 = tf.cond(phase_train, lambda: tf.nn.dropout(fc2, 0.5), lambda: fc2)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(fc2, n_classes, use_bias=True, kernel_initializer=xavier_init)

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


def vgg_13(input_shape, n_classes):
    """

    :param input_shape:
    :param n_classes:
    :return:
    """
    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')

    activation = tf.nn.relu
    # Block 1
    layer = tf.layers.conv2d(x, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 2
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 3
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 4
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 5
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # FC Layer
    flat_layer = tf.contrib.layers.flatten(layer)
    xavier_init = tf.initializers.variance_scaling(scale=1)

    with tf.variable_scope('fc1'):
        fc1 = tf.layers.dense(flat_layer, 4096, activation=activation, use_bias=True, kernel_initializer=xavier_init)
        fc1 = tf.cond(phase_train, lambda: tf.nn.dropout(fc1), lambda: fc1)
    with tf.variable_scope('fc2'):
        fc2 = tf.layers.dense(flat_layer, 4096, activation=activation, use_bias=True, kernel_initializer=xavier_init)
        fc2 = tf.cond(phase_train, lambda: tf.nn.dropout(fc2), lambda: fc2)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(fc2, n_classes, use_bias=True, kernel_initializer=xavier_init)

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


def vgg_16(input_shape, n_classes):
    """

    :param input_shape:
    :param n_classes:
    :return:
    """
    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')

    activation = tf.nn.relu
    # Block 1
    layer = tf.layers.conv2d(x, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 2
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 3
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 4
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 5
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    flat_layer = tf.contrib.layers.flatten(layer)
    xavier_init = tf.initializers.variance_scaling(scale=1)

    with tf.variable_scope('fc1'):
        fc1 = tf.layers.dense(flat_layer, 4096, activation=activation, use_bias=True, kernel_initializer=xavier_init)
        fc1 = tf.cond(phase_train, lambda: tf.nn.dropout(fc1), lambda: fc1)
    with tf.variable_scope('fc2'):
        fc2 = tf.layers.dense(flat_layer, 4096, activation=activation, use_bias=True, kernel_initializer=xavier_init)
        fc2 = tf.cond(phase_train, lambda: tf.nn.dropout(fc2), lambda: fc2)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(fc2, n_classes, use_bias=True, kernel_initializer=xavier_init)


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


def vgg_19(input_shape, n_classes):
    """

    :param input_shape:
    :param n_classes:
    :return:
    """
    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')

    activation = tf.nn.relu
    # Feature Extractor
    # Block 1
    layer = tf.layers.conv2d(x, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 2
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 3
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 4
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 5
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # FC Layers
    flat_layer = tf.contrib.layers.flatten(layer)
    xavier_init = tf.initializers.variance_scaling(scale=1)

    with tf.variable_scope('fc1'):
        fc1 = tf.layers.dense(flat_layer, 4096, activation=activation, use_bias=True, kernel_initializer=xavier_init)
        fc1 = tf.cond(phase_train, lambda: tf.nn.dropout(fc1), lambda: fc1)
    with tf.variable_scope('fc2'):
        fc2 = tf.layers.dense(flat_layer, 4096, activation=activation, use_bias=True, kernel_initializer=xavier_init)
        fc2 = tf.cond(phase_train, lambda: tf.nn.dropout(fc2), lambda: fc2)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(fc2, n_classes, use_bias=True, kernel_initializer=xavier_init)

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


def create_session(prefix):

    """config Option
     allow_soft_placement :  if cannot put a node in a gpu , put node to in a cpu
     log_device_placement :  show where each node is assigned
     config.gpu_options.allow_growth : 처음부터 메모리를 점유하지 말고 필요한 메모리를 점차 증가 시킵니다
    """

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    model_dir = './{}_models'.format(prefix)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    saver = tf.train.Saver()

    log_dir = './{}_logs'.format(prefix)
    tf_writer = tf.summary.FileWriter(log_dir)

    return sess, saver, tf_writer

def training(sess, n_step, batch_xs, batch_ys, ops):
    """
    Usage :
    >>> training(sess, n_step, batch_xs, batch_ys, ops)

    :param sess: tf.Session
    :param n_step: int | E.g)
    :param batch_xs: xs | E.g)
    :param batch_ys: ys | E.g)
    :param ops: tensor operations | E.g)
    :return: cost values
    """

    cost_values = []
    for i in range(n_step):
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