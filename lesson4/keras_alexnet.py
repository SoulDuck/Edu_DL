import __init__
from keras.models import Model
from keras.layers import Input , Dense , Conv2D , MaxPooling2D , Flatten
from keras.optimizers import SGD
import configure as cfg


def alexnet(x,datagen):
    """
    paper :
    Usage :
    >>> x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
    >>> dex = DogExtractor('/Users/seongjungkim/PycharmProjects/Edu_DL/data/dog_breed')
    >>> doggen = DogDataGenerator(dex)
    >>> alexnet()

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
    top_conv = Flatten()(conv5_layer)
    fc1 = Dense(4096, activation='relu')(top_conv)
    fc2 = Dense(4096, activation='relu')(fc1)
    pred = Dense(cfg.n_classes , activation='softmax')(fc2)
    model = Model(x,pred)
    # Training
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mse', 'accuracy'])
    model.fit_generator(generator=datagen)
