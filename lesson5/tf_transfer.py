import keras
from keras import models
from keras import layers


def pretrained_end2end(conv_base , fc_units_list ):
    model = models.Sequential()
    model.add(conv_base)
    conv_base.trainable = False
    model.add(layers.Flatten())


    for i,fc_units in enumerate(fc_units_list[:-1]):
        model.add(layers.Dense(fc_units , activation='relu'))
    model.add(layers.Dense(fc_units_list[-1] , activation='softmax'))
    model.summary()


