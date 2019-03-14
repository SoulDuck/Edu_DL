import numpy as np
import tensorflow as tf
from PIL import Image
import math
# Receptive field 을 계산해주는 calculator 입니다.

def pad_calculator(n_in, n_out, k, s):
    # n_out 이 나오기 위해 필요한 padding 의 양을 결정 합니다
    p = (s * (n_out -1 ) + k - n_in)/2
    return p

def outFromIn(conv, layerIn):

    # input
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]

    k = conv[0]  # kernel size
    s = conv[1]  #  stride
    p = conv[2]  # padding size

    #TODO 추가된 padding 은 몇?

    n_out = math.ceil((n_in + 2 * p - k) / s) + 1  # 요소가 부족해 Convolution 수행할수 없으면 Drop한다
    pos = (n_out - 1) * s - n_in + k  # position
    pos = pos / 2

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in

    start_out = start_in + ((k - 1) / 2 - pos ) * j_in
    return n_out, j_out, r_out, start_out

if __name__ == '__main__':

    # Fomular
    # Activation_map size
    img = Image.open('sample_image.png').resize((227, 227))
    np_img = np.asarray(img)
    convnet = [[11, 4, 0], [3, 2, 0], [5, 1, 0], [3, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0], [6, 1, 0],
               [1, 1, 0]]

    convnet = [[11, 4, 0], [3, 2, 0], [5, 1, 0], [3, 2, 0]]

    layer_names = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc6-conv', 'fc7-conv']
    imsize = 227

    # padding calculator
    # 추가된
    p = pad_calculator(n_in=5, n_out=5, k=3, s=2)
    print(p)
    curr_layer = [imsize, 1, 1, 0.5]
    # kernel , stride , padding
    print('input layer \n', curr_layer)
    for curr_conv in convnet:
        curr_layer = outFromIn(curr_conv, curr_layer)
        print(curr_layer)


    # Tensorflow
    n_filter = 3
    x = tf.placeholder(shape=[None, 227,227,1], dtype=tf.float32)
    layer = x
    for curr_conv in convnet:
        k = curr_conv [0]  # kernel size
        s = curr_conv [1]  # stride
        layer = tf.layers.conv2d(layer, n_filter, k, s, padding='valid')
        print(layer)
    x = tf.placeholder(shape=[None, 5, 5, 1], dtype=tf.float32)
    layer_same = tf.layers.conv2d(x, 1, 2, 2, padding='same')
    layer_valid = tf.layers.conv2d(x, 1, 2, 2, padding='valid')
    print(layer_same)
    print(layer_valid)