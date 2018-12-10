import tensorflow as tf

def alexnet(input_shape, n_classes):
    """

    :param input_shape: MUST be 4 dimension tensor | E.g) [None, 224,224,3]:
    :param n_classes:
    :return:
    """
    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

    # Convolution Initializer
    # Relu initalizizers
    he_init = tf.initializers.variance_scaling(scale=2)
    activation = tf.nn.relu
    # layer1
    with tf.variable_scope('conv1'):
        layer = tf.layers.conv2d(x, filters=96, kernel_size=11, strides=4, padding='same',
                                 kernel_initializer=he_init,
                                 activation=activation , use_bias=True)
        layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2,)
    # layer2
    with tf.variable_scope('conv2'):
        layer = tf.layers.conv2d(layer, filters=256, kernel_size=5, strides=1, padding='same',
                                 kernel_initializer=he_init,
                                 activation=activation, use_bias=True)
        layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2, padding='valid')
    # layer 3
    with tf.variable_scope('conv3'):
        layer = tf.layers.conv2d(layer, filters=384, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer=he_init,
                                 activation=activation, use_bias=True)
    # layer 4
    with tf.variable_scope('conv4'):
        layer = tf.layers.conv2d(layer, filters=384, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer=he_init,
                                 activation=activation, use_bias=True)
    # layer 5
    with tf.variable_scope('conv5'):
        layer = tf.layers.conv2d(layer, filters=256, kernel_size=3, strides=1, padding='same',
                                 kernel_initializer=he_init,
                                 activation=activation, use_bias=True)

    # Change node name
    top_conv = tf.identity(layer, 'top_conv')

    # Fully Connected Layers
    flat_layer = tf.layers.flatten(top_conv, name='flatten')

    # Fully Connected Layer Initializer
    with tf.variable_scope('fc1'):
        layer = tf.layers.dense(flat_layer, 4096, kernel_initializer=he_init, use_bias=True)
        layer = tf.cond(phase_train, lambda: tf.nn.dropout(layer, keep_prob), lambda: layer)
        layer = activation(layer)

    with tf.variable_scope('fc2'):
        layer = tf.layers.dense(layer, 4096, activation=activation, kernel_initializer=he_init, use_bias=True)
        layer = tf.cond(phase_train, lambda: tf.nn.dropout(layer, keep_prob), lambda: layer)
        layer = activation(layer)

    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

    # Mean cost values
    costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    cost_op = tf.reduce_mean(costs_op, name='cost_op')

    # Accuracy
    # 자동으로 tf.GraphKeys.METRIC tf.GraphKeys.METRIC_VARIABLES 에 추가됨
    #acc = tf.metrics.accuracy(labels=logits, predictions=y, name='acc_op')
    pred_cls = tf.argmax(logits, axis=1)
    y_cls = tf.argmax(y, axis=1)
    tf.reduce_mean(tf.cast(tf.equal(pred_cls, y_cls), dtype=tf.float32), name='acc_op')

    # Train op
    tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op, name='train_op')
