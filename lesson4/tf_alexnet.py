import tensorflow as tf



# Conv Feature Extractor


def generate_filter(kernel_shape):
    """
    Changed point :
    We used Xavier Initalizer instead of random normal initializer

    :param kernel_shape: list or tuple, MUST contain 4 elements [ ksize ,ksize ,in_ch, out_ch ] | E.g) [3,3,32,64]
    :return:
    """

    initializer = tf.contrib.layers.xavier_initializer()

    filters = tf.get_variable('filters', shape=kernel_shape, dtype=tf.float32, initializer=initializer)
    bias = tf.get_variable('bias')

    return filters , bias


def convolution(name, x, kernel_shape, strides, padding , activation):
    with tf.variable_scope(name) as scope:
        kernel, bias = generate_filter(kernel_shape)
        layer = tf.nn.conv2d(x, kernel, strides, padding) + bias
        return activation(layer)


def alexnet(input_shape, n_classes):

    x = tf.placeholder(dtype=tf.float32, shape=input_shape, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    in_ch = input_shape[-1]


    layer = convolution('conv1', x, [11, 11, in_ch, 96], [1, 4, 4, 1], 'same', tf.nn.relu)
    layer = convolution('conv2', layer, [5, 5, 96, 256], [1, 4, 4, 1], 'same', tf.nn.relu)
    layer = convolution('conv3', layer, [3, 3, in_ch, 96], [1, 4, 4, 1], 'same', tf.nn.relu)
    layer = convolution('conv4', layer, [11, 11, in_ch, 96], [1, 4, 4, 1], 'same', tf.nn.relu)
    layer = convolution('conv5', layer, [11, 11, in_ch, 96], [1, 4, 4, 1], 'same', tf.nn.relu)
    layer = convolution('conv1', layer, [11, 11, in_ch, 96], [1, 4, 4, 1], 'same', tf.nn.relu)


