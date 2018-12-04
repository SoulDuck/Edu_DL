import __init__
from keras.models import Model
from keras.layers import Input , Dense , Conv2D , MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD
import configure as cfg

def VGG_block(kernel_sizes , out_chs, strides, input):
    """
    VGG Block 은 아래와 같은 목적으로 설계되었습니다
    - filter size 을 (3,3) 으로 설계해 필터의 크기는 줄이고 ,
    층을 깊게 쌓아 receptive field 는 보전하고 Non-linearity 는 증가시키는 목적입니다


    :param kernel_sizes: tuple, E.g) (3,3)
    :param input: 입력으로 넣을수 있는 Tensor
    :return:
    """
    layer = input
    ksizes_outchs_strides = zip(kernel_sizes, out_chs, strides)
    for k , ch , s in ksizes_outchs_strides:
        layer = Conv2D(ch, (k, k), strides=(s, s), padding='same', activation='relu')(layer)
    layer = MaxPooling2D()(layer)
    return layer



def vgg11(x):
    """
    usage :
    >>> vgg11(x)

    :param x:
    :return:
    """
    layer = VGG_block(kernel_sizes=[3] , out_chs=[64], strides=[1] , input=x)
    layer = VGG_block(kernel_sizes=[3] , out_chs=[128], strides=[1] , input=layer)
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[256,256], strides=[1,1], input=layer)
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[512,512], strides=[1,1], input=layer)
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[512,512], strides=[1,1], input=layer)
    return layer

def vgg13(x):
    # VGG 13 Convnet
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[64,64], strides=[1,1] , input=x)
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[128,128], strides=[1,1] , input=layer)
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[256,256], strides=[1,1], input=layer)
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[512,512], strides=[1,1], input=layer)
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[512,512], strides=[1,1], input=layer)
    return layer

def vgg16():
    # VGG 16 Convnet
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[64,64], strides=[1,1] , input=x)
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[128,128], strides=[1,1] , input=layer )
    layer = VGG_block(kernel_sizes=[3,3,3] , out_chs=[256,256,256], strides=[1,1,1], input=layer)
    layer = VGG_block(kernel_sizes=[3,3,3] , out_chs=[512,512,512], strides=[1,1,1], input=layer)
    layer = VGG_block(kernel_sizes=[3,3,3] , out_chs=[512,512,512], strides=[1,1,1], input=layer)
    return layer

def vgg19():
    # VGG 19 Convnet
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[64,64], strides=[1,1] , input=x)
    layer = VGG_block(kernel_sizes=[3,3] , out_chs=[128,128], strides=[1,1] , input=layer )
    layer = VGG_block(kernel_sizes=[3,3,3,3] , out_chs=[256,256,256,256], strides=[1,1,1,1], input=layer)
    layer = VGG_block(kernel_sizes=[3,3,3,3] , out_chs=[512,512,512,512], strides=[1,1,1,1], input=layer)
    layer = VGG_block(kernel_sizes=[3,3,3,3] , out_chs=[512,512,512,512], strides=[1,1,1,1], input=layer)
    return layer

def logits(x ,n_classes):
    """
    usage :
    >>> top_conv = vgg11(x)
    >>> pred = logits(top_conv)

    :param x: 입력 tensor 입니다
    :return:
    """
    flat_top_conv = Flatten()(x)
    fc1 = Dense(4096, activation='relu')(flat_top_conv)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(4096, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)
    pred = Dense(n_classes, activation='softmax')(fc2)
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
