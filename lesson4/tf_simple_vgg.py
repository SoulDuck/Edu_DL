import tensorflow as tf


def vgg_11(input_shape, n_classes):
    """

    :param input_shape: list or tuple , MUST be 4 elements | E.g) [None , 32, 32, 3]
    :param n_classes: int | 110
    :return:
    """
    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')

    # Convolution Initializer
    he_init = tf.initializers.variance_scaling(scale=2)

    activation = tf.nn.relu
    # Block 1
    layer = tf.layers.conv2d(x, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 2
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 3
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 4
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 5
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # FC Layer
    # flat_layer = tf.layers.Flatten(layer)
    flat_layer = tf.contrib.layers.flatten(layer)
    xavier_init = tf.initializers.variance_scaling(scale=1)

    with tf.variable_scope('fc1'):
        layer = tf.layers.dense(flat_layer, 4096, activation=None, use_bias=True, kernel_initializer=xavier_init)
        layer = tf.layers.dropout(layer, keep_prob, training=phase_train)
        layer = activation(layer)
    with tf.variable_scope('fc2'):
        layer = tf.layers.dense(layer, 4096, activation=None, use_bias=True, kernel_initializer=xavier_init)
        layer = tf.layers.dropout(layer, keep_prob, training=phase_train)
        layer = activation(layer)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(layer, n_classes, use_bias=True, kernel_initializer=xavier_init)

    # Mean cost values
    costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    cost_op = tf.reduce_mean(costs_op, name='cost_op')

    # Accuracy
    # 자동으로 tf.GraphKeys.METRIC tf.GraphKeys.METRIC_VARIABLES 에 추가됨
    pred_cls = tf.argmax(logits, axis=1)
    y_cls = tf.argmax(y, axis=1)
    tf.reduce_mean(tf.cast(tf.equal(pred_cls, y_cls), dtype=tf.float32), name='acc_op')

    # Train op
    tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op, name='train_op')


def vgg_13(input_shape, n_classes):
    """

    :param input_shape:
    :param n_classes:
    :return:
    """
    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')

    # Convolution Initializer
    he_init = tf.initializers.variance_scaling(scale=2)

    activation = tf.nn.relu
    # Block 1
    layer = tf.layers.conv2d(x, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 2
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 3
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 4
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 5
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # FC Layer
    flat_layer = tf.contrib.layers.flatten(layer)
    xavier_init = tf.initializers.variance_scaling(scale=1)

    with tf.variable_scope('fc1'):
        layer = tf.layers.dense(flat_layer, 4096, activation=None, use_bias=True, kernel_initializer=xavier_init)
        layer = tf.layers.dropout(layer, keep_prob, training=phase_train)
        layer = activation(layer)
    with tf.variable_scope('fc2'):
        layer = tf.layers.dense(layer, 4096, activation=None, use_bias=True, kernel_initializer=xavier_init)
        layer = tf.layers.dropout(layer, keep_prob, training=phase_train)
        layer = activation(layer)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(layer, n_classes, use_bias=True, kernel_initializer=xavier_init)

    # Mean cost values
    costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    cost_op = tf.reduce_mean(costs_op, name='cost_op')

    # Accuracy
    # 자동으로 tf.GraphKeys.METRIC tf.GraphKeys.METRIC_VARIABLES 에 추가됨
    pred_cls = tf.argmax(logits, axis=1)
    y_cls = tf.argmax(y, axis=1)
    tf.reduce_mean(tf.cast(tf.equal(pred_cls, y_cls), dtype=tf.float32), name='acc_op')

    # Train op
    tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op, name='train_op')


def vgg_16(input_shape, n_classes):
    """

    :param input_shape:
    :param n_classes:
    :return:
    """

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')

    # Convolution Initializer
    he_init = tf.initializers.variance_scaling(scale=2)

    activation = tf.nn.relu
    # Block 1
    layer = tf.layers.conv2d(x, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 2
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 3
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 4
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 5
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    flat_layer = tf.contrib.layers.flatten(layer)
    xavier_init = tf.initializers.variance_scaling(scale=1)

    with tf.variable_scope('fc1'):
        layer = tf.layers.dense(flat_layer, 4096, activation=None, use_bias=True, kernel_initializer=xavier_init)
        layer = tf.layers.dropout(layer, keep_prob, training=phase_train)
        layer = activation(layer)
    with tf.variable_scope('fc2'):
        layer = tf.layers.dense(layer, 4096, activation=None, use_bias=True, kernel_initializer=xavier_init)
        layer = tf.layers.dropout(layer, keep_prob, training=phase_train)
        layer = activation(layer)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(layer, n_classes, use_bias=True, kernel_initializer=xavier_init)

    # Mean cost values
    costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    cost_op = tf.reduce_mean(costs_op, name='cost_op')

    # Accuracy
    # 자동으로 tf.GraphKeys.METRIC tf.GraphKeys.METRIC_VARIABLES 에 추가됨
    pred_cls = tf.argmax(logits, axis=1)
    y_cls = tf.argmax(y, axis=1)
    tf.reduce_mean(tf.cast(tf.equal(pred_cls, y_cls), dtype=tf.float32), name='acc_op')

    # Train op
    tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op, name='train_op')


def vgg_19(input_shape, n_classes):
    """

    :param input_shape:
    :param n_classes:
    :return:
    """
    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')

    # Convolution Initializer
    he_init = tf.initializers.variance_scaling(scale=2)

    activation = tf.nn.relu
    # Feature Extractor
    # Block 1
    layer = tf.layers.conv2d(x, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 64, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 2
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 128, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 3
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 256, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 4
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # Block 5
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.conv2d(layer, 3, 512, 1, padding='same', activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=he_init)
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2, padding='valid')

    # FC Layers
    flat_layer = tf.contrib.layers.flatten(layer)
    xavier_init = tf.initializers.variance_scaling(scale=1)

    with tf.variable_scope('fc1'):
        layer = tf.layers.dense(flat_layer, 4096, activation=None, use_bias=True, kernel_initializer=xavier_init)
        layer = tf.layers.dropout(layer, keep_prob, training=phase_train)
        layer = activation(layer)
    with tf.variable_scope('fc2'):
        layer = tf.layers.dense(layer, 4096, activation=None, use_bias=True, kernel_initializer=xavier_init)
        layer = tf.layers.dropout(layer, keep_prob, training=phase_train)
        layer = activation(layer)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(layer, n_classes, use_bias=True, kernel_initializer=xavier_init)

    # Mean cost values
    costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    cost_op = tf.reduce_mean(costs_op, name='cost_op')

    # Accuracy
    # 자동으로 tf.GraphKeys.METRIC tf.GraphKeys.METRIC_VARIABLES 에 추가됨
    pred_cls = tf.argmax(logits, axis=1)
    y_cls = tf.argmax(y, axis=1)
    tf.reduce_mean(tf.cast(tf.equal(pred_cls, y_cls), dtype=tf.float32), name='acc_op')

    # Train op
    tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op, name='train_op')
