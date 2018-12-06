import tensorflow as tf
import os


def alexnet(input_shape, n_classes):
    """

    :param input_shape: MUST be 4 dimension tensor | E.g) [None, 224,224,3]:
    :param n_classes:
    :return:
    """

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool)

    # Convolution Initializer
    he_init = tf.initializers.variance_scaling(scale=2)

    # layer1
    with tf.variable_scope('conv1'):
        layer = tf.layers.conv2d(x, filters=96, kernel_size=11, strides=4, padding='same',
                                 kernel_initializer=he_init,
                                 activation=tf.nn.relu, use_bias=True)
        layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2,)
    # layer2
    with tf.variable_scope('conv2'):
        layer = tf.layers.conv2d(layer, filters=256, kernel_size=5, strides=1, padding='same',
                                 kernel_initializer=he_init,
                                 activation=tf.nn.relu, use_bias=True)
        layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2, padding='valid')
    # layer 3
    with tf.variable_scope('conv3'):
        layer = tf.layers.conv2d(layer, filters=384, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer=he_init,
                                 activation=tf.nn.relu, use_bias=True)
    # layer 4
    with tf.variable_scope('conv4'):
        layer = tf.layers.conv2d(layer, filters=384, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer=he_init,
                                 activation=tf.nn.relu, use_bias=True)
    # layer 5
    with tf.variable_scope('conv5'):
        layer = tf.layers.conv2d(layer, filters=256, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer=he_init,
                                 activation=tf.nn.relu, use_bias=True)

    # Change node name
    top_conv = tf.identity(layer, 'top_conv')

    # Fully Connected Layers
    flat_layer = tf.layers.flatten(top_conv, name='flatten')

    # Fully Connected Layer Initializer
    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('fc1'):
        layer = tf.layers.dense(flat_layer, 4096, activation=tf.nn.relu, kernel_initializer=xavier_init, use_bias=True)

    with tf.variable_scope('fc2'):
        layer = tf.layers.dense(layer, 4096, activation=tf.nn.relu, kernel_initializer=xavier_init, use_bias=True)

    with tf.variable_scope('logits'):
        logits = tf.layers.dense(layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

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

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
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
