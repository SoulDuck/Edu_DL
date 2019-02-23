from keras.models import Model
from keras.layers import Input , Dense , Conv2D , MaxPooling2D , Flatten , Dropout
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam


def alexnet(input_shape, n_classes):
    """

    :param input_shape: tuple or list | Color Image : [height , widht , 3] ,Grey  Image : [height , widht , 1]
    :param n_classes: int | the number of classes
    :return:
    """
    # Input node
    x = Input(shape=input_shape)

    # Conv Layer 1
    conv1_layer = Conv2D(96, (11, 11), strides=(4, 4), padding='same', activation='relu')(x)
    conv1_layer = MaxPooling2D(pool_size=3, strides=2, padding='valid')(conv1_layer)

    # Conv Layer 2
    conv2_layer = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(conv1_layer)
    conv2_layer = MaxPooling2D(pool_size=3, strides=2, padding='valid')(conv2_layer)

    # Conv Layer 3
    conv3_layer = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_layer)
    conv3_layer = MaxPooling2D()(conv3_layer)

    # Conv Layer 4
    conv4_layer = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_layer)
    conv4_layer = MaxPooling2D()(conv4_layer)

    # Conv Layer 5
    conv5_layer = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv4_layer)

    # line up the layers
    flatten_layer = Flatten()(conv5_layer)

    # Fully Connected layer 1
    fc1 = Dense(4096, activation='relu')(flatten_layer)
    fc1 = Dropout(0.5)(fc1)

    # Fully Connected layer 2
    fc2 = Dense(4096, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    # Logits layer
    pred = Dense(n_classes, activation='softmax')(fc2)

    # Create Model
    model = Model(conv5_layer, pred)

    # show Model
    model.summary()
    return model


def training(model, optimizer_name, lr, epochs, data_generator):
    """
    >>> from extract import DogExtractor
    >>> from load import DogDataGenerator
    >>> dex = DogExtractor('../data/dog_breed')
    >>> doggen = DogDataGenerator(dex)
    >>> model = alexnet((224,224,3) , 120)
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
