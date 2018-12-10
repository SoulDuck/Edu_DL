import tensorflow as tf
import os
import sys

# Conv Feature Extractor
def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('{}_summaries'.format(name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def ops_summaries(ops):
    tf.summary.scalar('cost', ops['cost_op'])
    tf.summary.scalar('accuracy', ops['acc_op'])
    """
    if you want add image to tensorboard, uncomment this line 
    tf.summary.image('input', ops['x'], 16)
    """


def generate_filter(kernel_shape):
    """
    Changed point :
    We used Xavier Initalizer instead of random normal initializer

    :param kernel_shape: list or tuple, MUST contain 4 elements [ ksize ,ksize ,in_ch, out_ch ] | E.g) [3,3,32,64]
    :return:
    """
    n_out = kernel_shape[-1]
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    filters = tf.get_variable('filters', shape=kernel_shape, dtype=tf.float32, initializer=initializer)
    bias = tf.get_variable('bias', initializer=tf.constant(0.0, shape=[n_out]))
    variable_summaries('filters', filters)
    variable_summaries('bias', bias)

    return filters, bias


def convolution(name, x, kernel_shape, strides, padding, activation):
    with tf.variable_scope(name):
        kernel, bias = generate_filter(kernel_shape)
        layer = tf.nn.conv2d(x, kernel, strides, padding) + bias

        return activation(layer)


def generate_units(n_in_units, n_out_units):
    initializer = tf.contrib.layers.xavier_initializer()
    units = tf.get_variable('units', [n_in_units, n_out_units], tf.float32, initializer)
    bias = tf.get_variable('bias', [n_out_units], tf.float32, initializer)

    return units, bias


def fc(name, x, n_out_units, dropout_prob, phase_train, activation):
    """
    - Generate Fully Connected Layers Unit
    - Matrix multiply input with units and bias

    :param name: str | E.g) fc1
    :param x: tensor , MUST be tensor dimension 2 , | E.g)[2045, 10]
    :param n_out_units: int | E.g)1024
    :param dropout_prob: float | E.g) 0.5
    :param phase_train: type bool tensor( placeholer node )
    :param activation:
    :return: tensor with 2 dimension
    """

    n_in_units = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        units, bias = generate_units(n_in_units, n_out_units)
        layer = tf.matmul(x, units) + bias
        layer = tf.cond(phase_train, lambda: tf.nn.dropout(layer, dropout_prob), lambda: layer)

        if activation:
            layer = activation(layer)
        return layer


def alexnet(input_shape, n_classes):
    """

    :param input_shape: MUST be 4 dimension tensor | E.g) [None, 224,224,3]:
    :param n_classes:
    :return:
    """

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    phase_train = tf.placeholder(dtype=tf.bool)
    in_ch = input_shape[-1]

    # layer1

    layer = convolution('conv1', x, [11, 11, in_ch, 96], [1, 4, 4, 1], 'SAME', tf.nn.relu)
    layer = tf.nn.max_pool(layer, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # layer2
    layer = convolution('conv2', layer, [5, 5, 96, 256], [1, 1, 1, 1], 'SAME', tf.nn.relu)
    layer = tf.nn.max_pool(layer, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # layer 3
    layer = convolution('conv3', layer, [3, 3, 256, 384], [1, 1, 1, 1], 'SAME', tf.nn.relu)

    # layer 4
    layer = convolution('conv4', layer, [3, 3, 384, 384], [1, 1, 1, 1], 'SAME', tf.nn.relu)

    # layer 5
    layer = convolution('conv5', layer, [3, 3, 384, 256], [1, 1, 1, 1], 'SAME', tf.nn.relu)

    # Change node name
    top_conv = tf.identity(layer, 'top_conv')

    # Fully Connected Layers
    flat_layer = tf.contrib.layers.flatten(top_conv)

    # FC Layer 1
    layer = fc('fc1', flat_layer, 4096, 0.5, phase_train, tf.nn.relu)
    layer = fc('fc2', layer, 4096, 0.5, phase_train, tf.nn.relu)
    logits = fc('logits', layer, n_classes, 1.0, phase_train, None)

    # Probability
    preds = tf.nn.softmax(logits)

    # Mean cost values
    costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    cost_op = tf.reduce_mean(costs_op)

    # Accuracy
    preds_cls = tf.argmax(preds, axis=1)
    y_cls = tf.argmax(y, axis=1)
    acc_op = tf.reduce_mean(tf.cast(tf.equal(preds_cls, y_cls), tf.float32))

    # operations
    ops = {'x': x, 'y': y, 'phase_train': phase_train, 'cost_op': cost_op, 'acc_op': acc_op}

    # summary ops
    ops_summaries(ops)

    summaries_op = tf.summary.merge_all()
    ops['summaries_op'] = summaries_op

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


def create_saver(model_dir, max_to_keep = 10):
    """

    :param model_dir: str | E.g) : 'Alexnet'
    :param max_to_keep : int | E.g) : 10
    :return:
    """
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    return tf.train.Saver(max_to_keep=max_to_keep)


def create_logger(log_dir):
    """

    :param log_dir: str | E.g) : 'Alexnet'
    :return:
    """
    return tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())


def training(sess, loader, ops, writer, global_step, n_iter):
    """
    Usage :
    >>> training(sess, loader, ops, writer, global_step, n_iter)
    :param sess: tf.Session
    :param loader :  Class
    :param ops: tensor operations | E.g)
    :param writer: tf.Summary.Writer
    :param global_step: int | 1000
    :param n_iter: int | 100

    :return: cost values
    """

    step = 0
    for step in range(global_step, global_step + n_iter):
        sys.stdout.write('\r {} {}'.format(global_step, global_step + n_iter))
        sys.stdout.flush()
        batch_xs , batch_ys = loader.random_next_batch(onehot=True)
        fetches = [ops['train_op'], ops['cost_op'], ops['summaries_op']]
        feed_dict = {ops['x']: batch_xs, ops['y']: batch_ys, ops['phase_train']: True}
        _, cost, summeries = sess.run(fetches, feed_dict)
        writer.add_summary(summeries, step)
    return step


def eval(sess, batch_xs, batch_ys, ops, logger, global_step, saver=None, saver_name=None):
    """
    Usage :
    >>> eval(sess, batch_xs, batch_ys, ops, logger , global_step, saver)
    :param sess: tf.Session
    :param batch_xs: xs | E.g)
    :param batch_ys: ys | E.g)
    :param ops: tensor operations | E.g)
    :param logger: tf.Summary.Writer
    :param global_step: int | 1000

    :return: cost values
    """

    fetches = [ops['acc_op'], ops['cost_op'], ops['summaries_op']]
    feed_dict = {ops['x']: batch_xs, ops['y']: batch_ys, ops['phase_train']: False}
    acc, cost, summeries = sess.run(fetches, feed_dict)
    logger.add_summary(summeries, global_step=global_step)
    if saver:
        saver.save(saver_name, global_step=global_step)
