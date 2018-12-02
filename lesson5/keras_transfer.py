from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
import numpy as np
import sys

conv_base = VGG16(weights='imagenet' , include_top=False)
conv_base.summary()

"""
Keras Pretrained-Model List
1.
2.
3.
4.
5.
6.
7.
8.
9.
 
"""

def cache( conv_base, images , labels , batch_size, rescale=1. / 255):
    n_samples = len(labels)
    data_generator = ImageDataGenerator(rescale=rescale)
    batch_size = batch_size
    generator = data_generator.flow(images , labels , batch_size)
    features_list =[]
    labels = []
    for i,(inputs_batch , labels_batch) in enumerate(generator):
        sys.stdout.write('\r{}'.format(i))
        sys.stdout.flush()
        features_batch = conv_base.predict(inputs_batch)
        features_list.append(features_batch)
        labels.extend(labels_batch)
        if i >= int(n_samples / batch_size):
            break;
    features = np.vstack(features_list)
    del features_list
    return features , labels


def pretrained_end2end(conv_base , fc_units_list ):
    model = models.Sequential()
    model.add(conv_base)
    conv_base.trainable = False
    model.add(layers.Flatten())


    for i,fc_units in enumerate(fc_units_list[:-1]):
        model.add(layers.Dense(fc_units , activation='relu'))
    model.add(layers.Dense(fc_units_list[-1] , activation='softmax'))
    model.summary()


def finetuning():
    pass;


