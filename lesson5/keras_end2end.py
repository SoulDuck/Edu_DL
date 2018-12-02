import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


def pretrained_end2end(conv_base , fc_units_list ):
    model = models.Sequential()
    model.add(conv_base)
    conv_base.trainable = False
    model.add(layers.Flatten())


    for i,fc_units in enumerate(fc_units_list[:-1]):
        print(fc_units)
        model.add(layers.Dense(fc_units , activation='relu'))
    model.add(layers.Dense(fc_units_list[-1] , activation='softmax'))
    model.summary()
    return model


def pretrained_datagenerator(train_data, train_label, val_data, val_label, batch_size):
    # Train Data Generator
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_generator = train_datagen.flow(train_data, train_label, batch_size=batch_size)
    # Validation Data Generator
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow(val_data, val_label, batch_size=batch_size)

    return train_generator , val_generator


def pretrained_train(model , train_generator , val_generator ,  epochs=20 ):
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=epochs, validation_data=val_generator,
                                  validation_steps=50, verbose=2)
    return history

def pretrained_compile(model):
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model
