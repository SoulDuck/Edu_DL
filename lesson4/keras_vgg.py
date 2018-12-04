from keras.models import Model
from keras.layers import Input , Dense , Conv2D , MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam


def vgg_block(kernel_sizes, out_chs, strides, x):
    """
    VGG Block 은 아래와 같은 목적으로 설계되었습니다
    - filter size 을 (3,3) 으로 설계해 필터의 크기는 줄이고 ,
    층을 깊게 쌓아 receptive field 는 보전하고 Non-linearity 는 증가시키는 목적입니다

    :param kernel_sizes: list or tuple | E.g) [3, 3, 3, 3]
    :param out_chs: list or tuple | E.g) [256, 256, 256, 256]
    :param strides: list or tuple | E.g) [1, 1, 1, 1]
    :param x: tensor |
    :return:
    """

    layer = x
    ksizes_outchs_strides = zip(kernel_sizes, out_chs, strides)
    for k, ch, s in ksizes_outchs_strides:
        layer = Conv2D(ch, (k, k), strides=(s, s), padding='same', activation='relu')(layer)
    layer = MaxPooling2D()(layer)
    return layer


def fc_block(x, n_classes):
    """

    :param x: tensor
    :param n_classes: int
    :return:
    """
    flat_top_conv = Flatten()(x)
    fc1 = Dense(4096, activation='relu')(flat_top_conv)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(4096, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)
    pred = Dense(n_classes, activation='softmax')(fc2)
    return pred


def vgg_11(input_shape, n_classes):
    """
    Usage :
    >>> vgg_11(shape=(224,224,3) , n_classes=120)

    :param input_shape: tuple or list | E.g) (224,224,3)
    :param n_classes: int | E.g) 120
    :return: keras model
    """

    # Input Node
    x = Input(shape=input_shape)

    # VGG 13 Convnet
    layer = vgg_block(kernel_sizes=[3], out_chs=[64], strides=[1], input=x)
    layer = vgg_block(kernel_sizes=[3], out_chs=[128], strides=[1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[256, 256], strides=[1, 1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[512, 512], strides=[1, 1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[512, 512], strides=[1, 1], input=layer)

    # FC Layers
    pred = fc_block(layer, n_classes)

    # Keras model
    model = Model(x, pred)

    # Show model summarize
    model.summary()

    return model


def vgg_13(input_shape, n_classes):
    """
    Usage :
    >>> vgg_13(shape=(224,224,3) , n_classes=120)

    :param input_shape: tuple or list | E.g) (224,224,3)
    :param n_classes: int | E.g) 120
    :return: keras model
    """

    # Input Node
    x = Input(shape=input_shape)

    # VGG 13 Convnet
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[64, 64], strides=[1, 1], input=x)
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[128, 128], strides=[1, 1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[256, 256], strides=[1, 1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[512, 512], strides=[1, 1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[512, 512], strides=[1, 1], input=layer)

    # FC Layers
    pred = fc_block(layer, n_classes)

    # Keras model
    model = Model(x, pred)

    # Show model summarize
    model.summary()

    return model


def vgg_16(input_shape, n_classes):
    """
    Usage :
    >>> vgg_16(shape=(224,224,3) , n_classes=120)

    :param input_shape: tuple or list | E.g) (224,224,3)
    :param n_classes: int | E.g) 120
    :return: keras model
    """

    # Input node
    x = Input(shape=input_shape)

    # Feature Extractor
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[64, 64], strides=[1, 1], input=x)
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[128, 128], strides=[1, 1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3, 3], out_chs=[256, 256, 256], strides=[1, 1, 1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3, 3], out_chs=[512, 512, 512], strides=[1, 1, 1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3, 3], out_chs=[512, 512, 512], strides=[1, 1, 1], input=layer)

    # FC Layers
    pred = fc_block(layer, n_classes)

    # Keras model
    model = Model(x, pred)

    # Show model summarize
    model.summary()

    return model


def vgg_19(input_shape, n_classes):
    """
    Usage :
    >>> vgg_19(shape=(224,224,3) , n_classes=120)

    :param input_shape: tuple or list | E.g) (224,224,3)
    :param n_classes: int | E.g) 120
    :return: keras model
    """

    # Input node
    x = Input(shape=input_shape)

    # Feature Extractor
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[64, 64], strides=[1, 1], input=x)
    layer = vgg_block(kernel_sizes=[3, 3], out_chs=[128, 128], strides=[1, 1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3, 3, 3], out_chs=[256, 256, 256, 256], strides=[1, 1, 1, 1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3, 3, 3], out_chs=[512, 512, 512, 512], strides=[1, 1, 1, 1], input=layer)
    layer = vgg_block(kernel_sizes=[3, 3, 3, 3], out_chs=[512, 512, 512, 512], strides=[1, 1, 1, 1], input=layer)

    # FC Layers
    pred = fc_block(layer, n_classes)

    # Keras model
    model = Model(x, pred)

    # Show model summarize
    model.summary()

    return model


def training(model, optimizer_name, lr, epochs, data_generator):
    """
    # Usage :
    >>> dex = DogExtractor('../data/dog_breed')
    >>> doggen = DogDataGenerator(dex)
    >>> model = vgg_11((224,224,3) , 120)
    >>> training(model, 'momentum', 0.1, epochs=300, data_generator= doggen)

    :param model: keras model
    :param optimizer_name: str | E.g) 'sgd'
    :param lr: float | E.g) 0.001
    :param epochs: int | E.g) 100
    :param data_generator:  
    :return: history result
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
