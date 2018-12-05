import tensorflow as tf

# Conv Feature Extractor
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
    bias = tf.get_variable('bias', initializer=tf.constant(0, shape=[kernel_shape[n_out]]))

    return filters, bias


def convolution(name, x, kernel_shape, strides, padding, activation):
    with tf.variable_scope(name) as scope:
        kernel, bias = generate_filter(kernel_shape)
        layer = tf.nn.conv2d(x, kernel, strides, padding) + bias

        return activation(layer)


def fc(name, x, n_out, dropout_prob, phase_train, activation):
    """
    - Generate Fully Connected Layers Unit
    - Matrix multiply input with units and bias

    :param name: str | E.g) fc1
    :param x: tensor , MUST be tensor dimension 2 , | E.g)[2045, 10]
    :param n_out: int | E.g)1024
    :param dropout_prob: float | E.g) 0.5
    :param phase_train: type bool tensor( placeholer node )
    :param activation:
    :return: tensor with 2 dimension
    """

    initializer = tf.contrib.layers.xavier_initializer
    n_input = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        units = tf.get_variable('units', [n_input, n_out], tf.float32, initializer)
        bias = tf.get_variable('bias', [n_out], tf.float32, initializer)
        layer = tf.matmul(x, units) + bias
        layer = tf.cond(phase_train, lambda: tf.nn.dropout(layer, dropout_prob), layer)

        if activation:
            layer = activation(layer)
        return layer


def alexnet(input_shape, n_classes):
    """

    :param input_shape:
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
    layer = convolution('conv2', layer, [5, 5, 96, 256], [1, 4, 4, 1], 'SAME', tf.nn.relu)
    layer = tf.nn.max_pool(layer, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # layer 3
    layer = convolution('conv3', layer, [3, 3, 256, 384], [1, 4, 4, 1], 'SAME', tf.nn.relu)

    # layer 4
    layer = convolution('conv4', layer, [3, 3, 256, 384], [1, 4, 4, 1], 'SAME', tf.nn.relu)

    # layer 5
    layer = convolution('conv5', layer, [3, 3, 384, 384], [1, 4, 4, 1], 'SAME', tf.nn.relu)

    # Change node name
    top_conv = tf.identity(layer, 'top_conv')

    # Fully Connected Layers
    flat_layer = tf.contrib.layers.flatten(top_conv)

    # FC Layer 1
    layer = fc('fc1', flat_layer, 4096, 0.5, phase_train, tf.nn.relu)
    layer = fc('fc1', layer, 4096, 0.5, phase_train, tf.nn.relu)
    logits = fc('fc1', layer, 4096, 0.5, phase_train, None)

    return logits









