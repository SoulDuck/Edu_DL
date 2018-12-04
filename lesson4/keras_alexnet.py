import __init__
from keras.models import Model
from keras.layers import Input , Dense , Conv2D , MaxPooling2D , Flatten , Dropout
from keras.optimizers import SGD
import configure as cfg


def alexnet(x):
    """
    paper :
    Usage :
    >>> x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
    >>> alexnet(x)
    :return:
    """
    # Modeling
    conv1_layer = Conv2D(96,(11,11),strides=(4,4),padding='same',activation='relu')(x)
    conv1_layer = MaxPooling2D(pool_size=3, strides=2, padding='valid')(conv1_layer)
    conv2_layer = Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu')(conv1_layer)
    conv2_layer = MaxPooling2D(pool_size=3 , strides=2, padding='valid')(conv2_layer)
    conv3_layer = Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu')(conv2_layer)
    conv3_layer = MaxPooling2D()(conv3_layer)
    conv4_layer = Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu')(conv3_layer)
    conv4_layer = MaxPooling2D()(conv4_layer)
    conv5_layer = Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu')(conv4_layer)
    return conv5_layer

def logits(x):
    flatten_layer = Flatten()(x)
    fc1 = Dense(1024, activation='relu')(flatten_layer)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)
    pred = Dense(cfg.n_classes , activation='softmax')(fc2)
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
