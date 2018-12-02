from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import sys

conv_base = VGG16(weights='imagenet' , include_top=False)
conv_base.summary()

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
        if i == int(n_samples / batch_size)-1:
            break;
    features = np.vstack(features_list)
    del features_list
    return features , labels

def pretrained_dense_layer(fc_units_list):
    model = models.Sequential()
    for idx , fc_units in enumerate(fc_units_list[:-1]):
        model.add(layers.Dense(fc_units, activation='relu' , input_dim=7*7*512))
        model.add(layers.Dropout(0.5))
    # Last layer
    model.add(layers.Dense(fc_units_list[-1], activation='softmax'))
    return model

def pretrained_compile(model):
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


def pretrained_train(model , train_data, train_labels , epochs , batch_size , val_data , val_labels):
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
              validation_data=[val_data, val_labels])


def pretrained_end2end(conv_base , fc_units_list ):
    model = models.Sequential()
    model.add(conv_base)
    conv_base.trainable = False
    model.add(layers.Flatten())


    for i,fc_units in enumerate(fc_units_list[:-1]):
        model.add(layers.Dense(fc_units , activation='relu'))
    model.add(layers.Dense(fc_units_list[-1] , activation='softmax'))
    model.summary()





