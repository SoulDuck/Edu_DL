import unittest
import dog_breed
import numpy as np
from keras.applications import VGG16
import sys
sys.path.append('../')
sys.path.append('../../')
import keras_end2end

class TestKerasEnd2End(unittest.TestCase):

    def setUp(self):
        # Download Cifar
        dogs_extractor = dog_breed.Dog_Extractor('./dog_breed' , 2)
        self.test_imgs = dogs_extractor.imgs
        self.test_labs = dogs_extractor.labs

        print('images shape : {}'.format(np.shape(self.test_imgs)))
        print('labels shape : {}'.format(np.shape(self.test_labs)))

        self.conv_base = VGG16(weights='imagenet', include_top=False , input_shape=(255,255,3))
    def test_pretrained_end2end(self):
        model = keras_end2end.pretrained_end2end(self.conv_base, [1024, 120])

    def test_pretrained_datagenerator(self):
        train_generator, val_generator = keras_end2end.pretrained_datagenerator(self.test_imgs, self.test_labs,
                                                                                self.test_imgs, self.test_labs,
                                                                                batch_size=20)

    def test_pretrained_compile(self):
        model = keras_end2end.pretrained_end2end(self.conv_base, [1024, 120])
        model = keras_end2end.pretrained_compile(model)

    def test_pretrained_train(self):
        model = keras_end2end.pretrained_end2end(self.conv_base, [1024, 120])
        model = keras_end2end.pretrained_compile(model)
        train_generator, val_generator = keras_end2end.pretrained_datagenerator(self.test_imgs, self.test_labs,
                                                                                self.test_imgs, self.test_labs,
                                                                                batch_size=20)

        keras_end2end.pretrained_train(model, train_generator, val_generator, epochs=2)

    def tearDown(self):
        pass;
