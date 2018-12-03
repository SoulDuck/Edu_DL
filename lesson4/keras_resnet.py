import keras
from keras.models import Model
from keras.layers.merge import add
from keras.layer import Input , Activation , Dense, Flatten , Conv2D

def stem(input):

def residual_block(input , kernel_sizes , out_chs , strides):
    layer = input
    ksizes_outchs_strides = zip(kernel_sizes, out_chs, strides)

    for k, ch, s in ksizes_outchs_strides:
        layer = Conv2D(ch, (k, k), strides=(s, s), padding='same', activation='relu')(layer)

    return add(input , layer )

def residual_block_projection(input , kernel_sizes , out_chs , strides):
    """
    Feature Map Size 가 작아지는
    :param input:
    :param kernel_sizes:
    :param out_chs:
    :param strides:
    :return:
    """
    layer = input
    projection_input = Conv2D(out_chs[-1], (1, 1), strides=(1, 1), padding='same', activation='relu')(layer)
    ksizes_outchs_strides = zip(kernel_sizes, out_chs, strides)
    for k, ch, s in ksizes_outchs_strides:
        layer = Conv2D(ch, (k, k), strides=(s, s), padding='same', activation='relu')(layer)
    return add(projection_input , layer)


def bottlenect_block(input , ksizes , out_chs , strides):

def bottlenect_block_projection(input, ksizes, out_chs, strides):
