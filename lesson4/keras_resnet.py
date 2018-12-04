import __init__
import keras
from keras.models import Model
from keras.layers.merge import add
from keras.layers import Input , Activation , Dense, Flatten , Conv2D ,MaxPooling2D , GlobalAveragePooling2D
from keras.optimizers import SGD

"""
목적 : resnet paper 에 구현된 model 을 따라 만들어 보고 결과를 확인합니다.
"""


######################################
######       Define Block     ########
######################################
def stem(input):
    layer = Conv2D(64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu')(input)
    layer = MaxPooling2D(pool_size=3 , strides=2)(layer)
    return layer

def residual_block(input, out_ch, kernel_size=(3,3)):
    """
    residual_block_projection 은 아래와 같은 특징을 가지고 있습니다
    - shortcut 연결 을 사용함으로서 학습 속도를 빠르게 합니다
    - 입력과 마지막 layer 을 Element-wise Add 을 합니다

    :param input: 입력되는 Feature map 입니다 , Tensor 형태여야 합니다
    :param out_ch: 최종적으로 출력되는 Channel 의 갯수입니다
    :param kernel_size: Input Layer , 첫번째 layer 에 적용되는 kernel(Filter) 크기 입니다.
    :return:
    """

    strides = (1, 1)
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(input)
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(layer)
    return add([input , layer])

def residual_block_projection(input, out_ch, kernel_size=(3,3), strides=(2,2)):
    """
    residual_block_projection 은 아래와 같은 특징을 가지고 있습니다

    - Feature map 의 size 가 줄어들고 Channel 이 늘어납니다.
    - Input shortcut 과 최종 layer Channel 의 갯수를 맞추어 주기 위해 , 1x1 Convolution NN 을 사용합니다
    - 입력과 마지막 layer 을 Element-wise Add 을 합니다

    :param input: 입력되는 Feature map 입니다 , Tensor 형태여야 합니다
    :param out_ch: 최종적으로 출력되는 Channel 의 갯수입니다
    :param kernel_size: Input Layer , 첫번째 layer 에 적용되는 kernel(Filter) 크기 입니다.
    :param strides:
    :return:
    """

    projection_input  = Conv2D(out_ch, (1,1), strides=strides, padding='same', activation='relu')(input)
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(input)
    layer = Conv2D(out_ch, kernel_size, strides=(1,1), padding='same', activation='relu')(layer)
    return add([projection_input , layer])

def bottlenect_block(input, out_ch, kernel_size=(3,3)):
    """
    bottlenect_block 은 아래와 같은 특징을 가지고 있습니다
    - 동일한 Level 의 resnet block 과 inferencing 할 때 걸리는 시간이 동일합니다
    - Feature map 이 더 많은 Non-linearity 을 통과함으로서 더 강력한 Feature Extracter 을 만들수 있습니다
    - bottlenect 을 구성합으로 필요한 Feature 을 추출할 수 있는 Feature Extraction 기능이 강화 됩니다
    - 입력과 마지막 layer 을 Element-wise Add 을 합니다

    :param input: 입력되는 Feature map 입니다 , Tensor 형태여야 합니다
    :param out_ch: 최종적으로 출력되는 Channel 의 갯수입니다
    :param kernel_size: 중간 layer 에 적용되는 kernel(Filter) 크기 입니다.
    :return:
    """
    strides=(1,1)
    layer = Conv2D(out_ch, (1,1), strides=strides, padding='same', activation='relu')(input)
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(layer)
    layer = Conv2D(out_ch * 4 , (1,1), strides=strides, padding='same', activation='relu')(layer)
    return add([input , layer])

def bottlenect_block_projection(input, out_ch, kernel_size=(3,3), strides=(2,2)):
    """
    bottlenect_block_projection은 3가지 목적이 있습니다

    - Feature map의 size 을 1/2 로 축소 합니다
    - Channel 의 수를 x2 합니다
    - 입력과 마지막 layer 을 Element-wise Add 을 합니다

    이를 수행하기 위해
    입력(input) 을 1x1 conv , out_ch * 4 로 Convolution 합니다(이를 Projection 이라 합니다)
    그리고 최종 layer 와 Projection Input 을 더합니다

    :param input: 입력되는 Feature map 입니다 , Tensor 형태여야 합니다
    :param out_ch: 최종적으로 출력되는 Channel 의 갯수입니다
    :param kernel_size: 중간 layer 에 적용되는 kernel(Filter) 크기 입니다.
    :param strides: Input 과 첫번째 Convolution layer 에 적용될 Strides 입니다
    :return:
    """

    projection_input = Conv2D(out_ch*4, (1, 1), strides=strides, padding='same', activation='relu')(input)
    layer = Conv2D(out_ch, kernel_size=(1,1), strides=strides, padding='same', activation='relu')(input)
    layer = Conv2D(out_ch, kernel_size=kernel_size, strides=(1,1), padding='same', activation='relu')(layer)
    layer = Conv2D(out_ch * 4 , kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(layer)

    return add([projection_input , layer])


######################################
######      Define resnet     ########
######################################

def resnet18(input):
    """
    Usage :
    >>> x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
    >>> dex = DogExtractor('/Users/seongjungkim/PycharmProjects/Edu_DL/data/dog_breed')
    >>> doggen = DogDataGenerator(dex)
    >>> resnet18(x , cfg.n_classes)

    :param input: input tensor
    :param n_classes: int , 몇 개의 클래스로 나뉘어 지는지
    :return:
    """

    layer=stem(input)
    # Block A
    layer = residual_block(layer, 64)
    layer = residual_block(layer, 64)

    # Block B
    layer = residual_block_projection(layer, 128, strides=(2,2))
    layer = residual_block(layer, 128)

    # Block C
    layer = residual_block_projection(layer, 256 , strides=(2,2))
    layer = residual_block(layer, 256)

    # Block D
    layer = residual_block_projection(layer, 512, strides=(2,2))
    layer = residual_block(layer, 512)

    layer = GlobalAveragePooling2D()(layer)
    pred = Dense(n_classes, activation='softmax')(layer)
    return pred

def resnet_34(input):
    """
    >>> x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
    >>> dex = DogExtractor('/Users/seongjungkim/PycharmProjects/Edu_DL/data/dog_breed')
    >>> doggen = DogDataGenerator(dex)
    >>> resnet34(x , cfg.n_classes)

    :param input:
    :param n_classes:
    :return:
    """

    layer = stem(input)
    # Block A
    layer = residual_block(layer, 64)
    layer = residual_block(layer, 64)
    layer = residual_block(layer, 64)

    # Block B
    layer = residual_block_projection(layer, 128, strides=(2,2))
    layer = residual_block(layer, 128)
    layer = residual_block(layer, 128)
    layer = residual_block(layer, 128)

    # Block C
    layer = residual_block_projection(layer, 256, strides=(2,2))
    layer = residual_block(layer, 256)
    layer = residual_block(layer, 256)
    layer = residual_block(layer, 256)
    layer = residual_block(layer, 256)
    layer = residual_block(layer, 256)

    # Block D
    layer = residual_block_projection(layer, 512, strides=(2,2))
    layer = residual_block(layer, 512)
    layer = residual_block(layer, 512)

    layer = GlobalAveragePooling2D()(layer)
    pred = Dense(n_classes, activation='softmax')(layer)
    return pred



# Resnet 50
def resnet_50(input):
    """
    Usage :
    >>> x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
    >>> dex = DogExtractor('/Users/seongjungkim/PycharmProjects/Edu_DL/data/dog_breed')
    >>> doggen = DogDataGenerator(dex)
    >>> resnet50(x , cfg.n_classes)

    :param input:
    :param n_classes:
    :return:
    """

    layer = stem(input)
    # Block A
    layer = bottlenect_block_projection(layer, out_ch=64 , strides=(1,1))
    layer = bottlenect_block(layer, out_ch=64)
    layer = bottlenect_block(layer, out_ch=64)

    # Block B
    layer = bottlenect_block_projection(layer, out_ch=128 , strides=(2,2))
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)

    # Block C
    layer = bottlenect_block_projection(layer, out_ch=256, strides=(2,2))
    layer = bottlenect_block(layer, out_ch=256)
    layer = bottlenect_block(layer, out_ch=256)
    layer = bottlenect_block(layer, out_ch=256)
    layer = bottlenect_block(layer, out_ch=256)
    layer = bottlenect_block(layer, out_ch=256)

    # Block D
    layer = bottlenect_block_projection(layer, out_ch=512, kernel_size=(2, 2))
    layer = bottlenect_block(layer, out_ch=512)
    layer = bottlenect_block(layer, out_ch=512)

    # Global Average Pooling
    layer = GlobalAveragePooling2D()(layer)
    pred = Dense(n_classes, activation='softmax')(layer)
    return pred



# Resnet 101
def resnet_101(input):
    """
    Usage :
    >>> x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
    >>> dex = DogExtractor('/Users/seongjungkim/PycharmProjects/Edu_DL/data/dog_breed')
    >>> doggen = DogDataGenerator(dex)
    >>> resnet101(x , cfg.n_classes)

    :param input:
    :param n_classes:
    :return:
    """

    layer = stem(input)
    # Block A
    layer = bottlenect_block_projection(layer, out_ch=64, strides=(1, 1))
    layer = bottlenect_block(layer, out_ch=64)
    layer = bottlenect_block(layer, out_ch=64)

    # Block B
    layer = bottlenect_block_projection(layer, out_ch=128, strides=(2, 2))
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)

    # Block C
    layer = bottlenect_block_projection(layer, out_ch=256, strides=(2, 2))
    for i in range(22):
        layer = bottlenect_block(layer, out_ch=256)

    # Block D
    layer = residual_block_projection(layer, 512, strides=(2, 2))
    layer = bottlenect_block(layer, out_ch=512)
    layer = bottlenect_block(layer, out_ch=512)
    # Global Average Pooling
    layer = GlobalAveragePooling2D()(layer)
    pred = Dense(n_classes, activation='softmax')(layer)
    return pred


def resnet_152(input):
    """
    Usage :
    >>> x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
    >>> dex = DogExtractor('/Users/seongjungkim/PycharmProjects/Edu_DL/data/dog_breed')
    >>> doggen = DogDataGenerator(dex)
    >>> resnet152(x , cfg.n_classes)

    :param input:
    :param n_classes:
    :return:
    """

    layer = stem(input)

    # Block A
    layer = bottlenect_block_projection(layer, out_ch=64, strides=(1, 1))
    layer = bottlenect_block(layer, out_ch=64)
    layer = bottlenect_block(layer, out_ch=64)

    # Block B
    layer = bottlenect_block_projection(layer, out_ch=128, strides=(2, 2))
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)

    # Block C
    layer = bottlenect_block_projection(layer, out_ch=256, strides=(2, 2))
    for i in range(35):
        layer = bottlenect_block(layer, out_ch=256)

    # Block D
    layer = bottlenect_block_projection(layer, out_ch=512, strides=(2, 2))
    layer = bottlenect_block(layer, out_ch=512)
    layer = bottlenect_block(layer, out_ch=512)


def logits(x , n_classes):
    """
    Usage :
    :param x:
    :param n_classes:
    :return:
    """
    # Global Average Pooling
    layer = GlobalAveragePooling2D()(x)
    pred = Dense(n_classes, activation='softmax')(layer)
    return pred

def training(x, pred, datagen , lr , epochs):
    """
    >>> dex = DogExtractor('../data/dog_breed')
    >>> doggen = DogDataGenerator(dex)
    >>> x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
    >>> top_conv = alexnet(x)
    >>> pred = logits(top_conv)
    >>> training(x, pred , datagen)

    :param model:
    :param datagen:
    :return:
    """
    # Training
    model = Model(x, pred)
    model.summary()
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mse', 'accuracy'])
    model.fit_generator(generator=datagen , epochs = epochs)
