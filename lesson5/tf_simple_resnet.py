import tensorflow as tf


def batch_normalization(x, phase_train, scope_name):
    with tf.variable_scope(scope_name):
        n_out = int(x.get_shape()[-1])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
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
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

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
    h, w = top_conv.get_shape()[1:3]
    gap_layer = tf.layers.average_pooling2d(top_conv, pool_size=(h, w), strides=(1, 1))

    # Flatten layer
    flat_layer = tf.contrib.layers.flatten(gap_layer)

    # logits
    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(flat_layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

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


def resnet_34(input_shape, n_classes):

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

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
    h, w = top_conv.get_shape()[1:3]
    gap_layer = tf.layers.average_pooling2d(top_conv, pool_size=(h, w), strides=(1, 1))

    # Flatten layer
    flat_layer = tf.contrib.layers.flatten(gap_layer)

    # logits
    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(flat_layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

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

    # Add trainable node to summary


def resnet_50(input_shape, n_classes):

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

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
    h, w = top_conv.get_shape()[1:3]
    gap_layer = tf.layers.average_pooling2d(top_conv, pool_size=(h, w), strides=(1, 1))

    # Flatten layer
    flat_layer = tf.contrib.layers.flatten(gap_layer)

    # logits
    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(flat_layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

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


def resnet_101(input_shape, n_classes):

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

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
    h, w = top_conv.get_shape()[1:3]
    gap_layer = tf.layers.average_pooling2d(top_conv, pool_size=(h, w), strides=(1, 1))

    # Flatten layer
    flat_layer = tf.contrib.layers.flatten(gap_layer)

    # logits
    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(flat_layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

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


def resnet_151(input_shape, n_classes):

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

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
    h, w = top_conv.get_shape()[1:3]
    gap_layer = tf.layers.average_pooling2d(top_conv, pool_size=(h, w), strides=(1, 1))

    # Flatten layer
    flat_layer = tf.contrib.layers.flatten(gap_layer)

    # logits
    xavier_init = tf.initializers.variance_scaling(scale=1)
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(flat_layer, n_classes, activation=None, kernel_initializer=xavier_init, use_bias=True)

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
