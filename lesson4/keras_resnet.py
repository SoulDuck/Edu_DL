import keras
from keras.models import Model
from keras.layers.merge import add
from keras.layer import Input , Activation , Dense, Flatten , Conv2D ,MaxPooling2D
import configure as cfg
"""
ref : 
"""
def stem(input):
    layer = Conv2D(64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu')(input)
    layer = MaxPooling2D(kernel_size=(3,3) , strides=(2,2))(layer)
    return layer

def residual_block(input, out_ch, kernel_size=(3,3), strides=(1,1)):
    """

    :param input:
    :param out_ch:
    :param kernel_size:
    :param strides:
    :return:
    """

    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(input)
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(layer)
    return add(input , layer )

def residual_block_projection(input, out_ch, kernel_size=(3,3), strides=(1,1)):
    """
    Feature map 의 size 가 줄어들면 Channel 이 늘어납니다.
    이때 shortcut 의 Channel 의 갯수를 맞추어 주기 위해 Projection Block 을 사용합니다
    :param input:
    :param kernel_sizes:
    :param out_chs:
    :param strides:
    :return:
    """
    projection_input  = Conv2D(out_ch, (1,1), strides=strides, padding='same', activation='relu')(input)
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(input)
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(layer)
    return add(projection_input , layer)

def bottlenect_block(input, out_ch, kernel_size=(3,3), strides=(1,1)):
    """

    :param input:
    :param out_ch:
    :param kernel_size:
    :param strides:
    :return:
    """
    layer = Conv2D(out_ch, (1,1), strides=strides, padding='same', activation='relu')(input)
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(layer)
    layer = Conv2D(out_ch * 4 , (1,1), strides=strides, padding='same', activation='relu')(layer)
    return add(input , layer)

def bottlenect_block_projection(input, out_ch, kernel_size=(3,3), strides=(1,1)):
    """

    :param input:
    :param out_ch:
    :param kernel_size:
    :param strides:
    :return:
    """
    projection_input = Conv2D(out_ch, (1, 1), strides=strides, padding='same', activation='relu')(input)

    layer = Conv2D(out_ch, (1,1), strides=strides, padding='same', activation='relu')(input)
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(layer)
    layer = Conv2D(out_ch * 4 , (1,1), strides=strides, padding='same', activation='relu')(layer)
    return add(projection_input , layer)


def resnet18(input):
    layer=stem(input)
    # Block A
    layer = residual_block(layer, 64)
    layer = residual_block(layer, 64)

    # Block B
    layer = residual_block_projection(layer, 128)
    layer = residual_block(layer, 128)

    # Block C
    layer = residual_block_projection(layer, 256)
    layer = residual_block(layer, 256)

    # Block D
    layer = residual_block_projection(layer, 512)
    layer = residual_block(layer, 512)

    return layer

def resnet_34(input):
    layer = stem(input)
    # Block A
    layer = residual_block(layer, 64)
    layer = residual_block(layer, 64)
    layer = residual_block(layer, 64)

    # Block B
    layer = residual_block_projection(layer, 128)
    layer = residual_block(layer, 128)
    layer = residual_block(layer, 128)
    layer = residual_block(layer, 128)

    # Block C
    layer = residual_block_projection(layer, 256)
    layer = residual_block(layer, 256)
    layer = residual_block(layer, 256)
    layer = residual_block(layer, 256)
    layer = residual_block(layer, 256)
    layer = residual_block(layer, 256)


    # Block D
    layer = residual_block_projection(layer, 512)
    layer = residual_block(layer, 512)
    layer = residual_block(layer, 512)

    return layer


# Resnet 50
def resnet_50(input):
    layer = stem(input)
    # Block A
    bottlenect_block(layer, out_ch=64, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=64, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=64, kernel_size=(3, 3))
    # Block B
    bottlenect_block_projection(layer, out_ch=128, kernel_size=(3,3))
    bottlenect_block(layer, out_ch=128, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=128, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=128, kernel_size=(3, 3))
    # Block C
    bottlenect_block_projection(layer, out_ch=256, kernel_size=(3,3))
    bottlenect_block(layer, out_ch=256, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=256, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=256, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=256, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=256, kernel_size=(3, 3))
    # Block D
    bottlenect_block_projection(layer, out_ch=512, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=512, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=512, kernel_size=(3, 3))



# Resnet 101
def resnet_101(input):
    layer = stem(input)
    # Block A
    bottlenect_block(layer, out_ch=64, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=64, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=64, kernel_size=(3, 3))
    # Block B
    bottlenect_block_projection(layer, out_ch=128, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=128, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=128, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=128, kernel_size=(3, 3))
    # Block C
    bottlenect_block_projection(layer, out_ch=256, kernel_size=(3, 3))
    for i in range(22):
        layer = bottlenect_block(layer, out_ch=256, kernel_size=(3, 3))
    # Block D
    bottlenect_block_projection(layer, out_ch=512, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=512, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=512, kernel_size=(3, 3))

def resnet_152(input):
    layer = stem(input)

    # Block A
    bottlenect_block(layer, out_ch=64, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=64, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=64, kernel_size=(3, 3))

    # Block B
    bottlenect_block_projection(layer, out_ch=128, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=128, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=128, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=128, kernel_size=(3, 3))

    # Block C
    bottlenect_block_projection(layer, out_ch=256, kernel_size=(3, 3))
    for i in range(35):
        layer = bottlenect_block(layer, out_ch=256, kernel_size=(3, 3))

    # Block D
    bottlenect_block_projection(layer, out_ch=512, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=512, kernel_size=(3, 3))
    bottlenect_block(layer, out_ch=512, kernel_size=(3, 3))