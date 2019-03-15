from keras.models import Model
from keras.layers.merge import add
from keras.layers import Input , Activation , Dense, Flatten , Conv2D ,MaxPooling2D , GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.optimizers import SGD

"""
resnet paper : 에 구현된 model 을 구현합니다

"""

def stem(x):
    layer = Conv2D(64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu')(x)
    layer = MaxPooling2D(pool_size=3 , strides=2)(layer)
    return layer


def residual_block(x, out_ch):
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
    :return: tensor
    """
    kernel_size = (3, 3)
    strides = (1, 1)
    # Plain layers
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(x)
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(layer)

    return add([x, layer])


def residual_block_projection(x, out_ch):
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
    :return: tensor
    """

    kernel_size = (3, 3)
    strides = (2, 2)
    # projection layers
    projection_input = Conv2D(out_ch, (1, 1), strides=strides, padding='same', activation='relu')(x)
    # Plain layers
    layer = Conv2D(out_ch, kernel_size, strides=strides, padding='same', activation='relu')(x)
    layer = Conv2D(out_ch, kernel_size, strides=(1, 1), padding='same', activation='relu')(layer)

    return add([projection_input, layer])


def bottlenect_block(x, out_ch):
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
    :return: tensor
    """

    kernel_size = (3, 3)
    strides = (1, 1)

    # Plain Layers
    layer = Conv2D(int(out_ch // 4), (1, 1), strides=strides, padding='same', activation='relu')(x)
    layer = Conv2D(int(out_ch // 4), kernel_size, strides=strides, padding='same', activation='relu')(layer)
    layer = Conv2D(out_ch, (1, 1), strides=strides, padding='same', activation='relu')(layer)
    return add([x, layer])


def bottlenect_block_projection(x, out_ch):
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
    :return: tensor
    """
    kernel_size = (3, 3)
    strides = (2, 2)

    # projection layer
    projection_input = Conv2D(out_ch, (1, 1), strides=strides, padding='same', activation='relu')(x)

    # Plain Layers
    layer = Conv2D(out_ch // 4, kernel_size=(1, 1), strides=strides, padding='same', activation='relu')(x)
    layer = Conv2D(out_ch // 4, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(layer)
    layer = Conv2D(out_ch, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(layer)

    return add([projection_input, layer])


def resnet_18(input_shape, n_classes):
    """
    Usage :
    >>> resnet_18(shape=(224,224,3) , n_classes=120)

    :param input_shape: tuple or list | E.g) (224,224,3)
    :param n_classes: int | E.g) 120
    :return: keras model
    """

    # Input
    x = Input(shape=input_shape)

    # Stem
    layer = stem(x)

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

    # FC layer
    layer = GlobalAveragePooling2D()(layer)

    # Logits
    pred = Dense(n_classes, activation='softmax')(layer)

    # keras model
    model = Model(x, pred)
    model.summary()

    return model


def resnet_34(input_shape, n_classes):
    """
    Usage :
    >>> resnet_34(shape=(224,224,3) , n_classes=120)
    :param input_shape: tuple or list | E.g) (224,224,3)
    :param n_classes: int | E.g) 120
    :return: keras model
    """
    # Input layer
    x = Input(shape=input_shape)

    # Stem
    layer = stem(x)

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

    # FC layer
    layer = GlobalAveragePooling2D()(layer)

    # Logits
    pred = Dense(n_classes, activation='softmax')(layer)

    # keras model
    model = Model(x, pred)
    model.summary()

    return model


def resnet_50(input_shape, n_classes):
    """
    Usage :
    >>> resnet_50(shape=(224,224,3) , n_classes=120)

    :param input_shape: tuple or list | E.g) (224,224,3)
    :param n_classes: int | E.g) 120
    :return: keras model
    """

    # Input Layer
    x = Input(shape=input_shape)

    # Stem
    layer = stem(x)

    # Block A
    layer = bottlenect_block_projection(layer, out_ch=64)
    layer = bottlenect_block(layer, out_ch=64)
    layer = bottlenect_block(layer, out_ch=64)

    # Block B
    layer = bottlenect_block_projection(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)

    # Block C
    layer = bottlenect_block_projection(layer, out_ch=256)
    layer = bottlenect_block(layer, out_ch=256)
    layer = bottlenect_block(layer, out_ch=256)
    layer = bottlenect_block(layer, out_ch=256)
    layer = bottlenect_block(layer, out_ch=256)
    layer = bottlenect_block(layer, out_ch=256)

    # Block D
    layer = bottlenect_block_projection(layer, out_ch=512)
    layer = bottlenect_block(layer, out_ch=512)
    layer = bottlenect_block(layer, out_ch=512)

    # FC layer
    layer = GlobalAveragePooling2D()(layer)

    # Logits
    pred = Dense(n_classes, activation='softmax')(layer)

    # keras model
    model = Model(x, pred)
    model.summary()

    return model


def resnet_101(input_shape, n_classes):
    """
    Usage :
    >>> resnet_101(shape=(224,224,3) , n_classes=120)

    :param input_shape: tuple or list | E.g) (224,224,3)
    :param n_classes: int | E.g) 120
    :return: keras model
    """

    # Input Layer
    x = Input(shape=input_shape)

    # Stem
    layer = stem(x)

    # Block A
    layer = bottlenect_block_projection(layer, out_ch=64)
    layer = bottlenect_block(layer, out_ch=64)
    layer = bottlenect_block(layer, out_ch=64)

    # Block B
    layer = bottlenect_block_projection(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)

    # Block C
    layer = bottlenect_block_projection(layer, out_ch=256)
    for i in range(22):
        layer = bottlenect_block(layer, out_ch=256)

    # Block D
    layer = residual_block_projection(layer, 512)
    layer = bottlenect_block(layer, out_ch=512)
    layer = bottlenect_block(layer, out_ch=512)

    # FC layer
    layer = GlobalAveragePooling2D()(layer)

    # Logits
    pred = Dense(n_classes, activation='softmax')(layer)

    # keras model
    model = Model(x, pred)
    model.summary()

    return model


def resnet_152(input_shape, n_classes):
    """
    Usage :
    >>> resnet_152(shape=(224,224,3) , n_classes=120)


    :param input_shape: tuple or list | E.g) (224,224,3)
    :param n_classes: int | E.g) 120
    :return: keras model
    """

    # Input Layer
    x = Input(shape=input_shape)

    # Stem
    layer = stem(x)

    # Block A
    layer = bottlenect_block_projection(layer, out_ch=64)
    layer = bottlenect_block(layer, out_ch=64)
    layer = bottlenect_block(layer, out_ch=64)

    # Block B
    layer = bottlenect_block_projection(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)
    layer = bottlenect_block(layer, out_ch=128)

    # Block C
    layer = bottlenect_block_projection(layer, out_ch=256)
    for i in range(35):
        layer = bottlenect_block(layer, out_ch=256)

    # Block D
    layer = bottlenect_block_projection(layer, out_ch=512)
    layer = bottlenect_block(layer, out_ch=512)
    layer = bottlenect_block(layer, out_ch=512)

    # FC layer
    layer = GlobalAveragePooling2D()(layer)

    # Logits
    pred = Dense(n_classes, activation='softmax')(layer)

    # keras model
    model = Model(x, pred)
    model.summary()

    return model


def training(model, optimizer_name, lr, epochs, data_generator):
    """
    >>> from DataExtractor.extract import DogExtractor
    >>> from DataExtractor.load import DogDataGenerator
    >>> dex = DogExtractor('../data/dog_breed')
    >>> doggen = DogDataGenerator(dex)
    >>> model = resnet_18((224,224,3) , 120)
    >>> training(model, 'momentum', 0.1, epochs=300, data_generator= doggen)

    :param model: keras model
    :param optimizer_name: str | E.g) 'sgd'
    :param lr: float | E.g)0.01
    :param epochs: int | E.g) 100
    :param data_generator: generator
    :return: keras history
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'sgd':
        optimizer = SGD(lr=lr, decay=1e-6, momentum=0.0, nesterov=False)

    elif optimizer_name == 'momentum':
        optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    elif optimizer_name == 'adagrad':
        optimizer = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    elif optimizer_name == 'adadelta':
        optimizer = Adadelta(lr=0.01, epsilon=None, decay=0.0)

    elif optimizer_name == 'adam':
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    elif optimizer_name == 'adamax':
        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    elif optimizer_name == 'nadam':
        optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    else:
        raise ValueError

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])
    return model.fit_generator(generator=data_generator, epochs=epochs)
