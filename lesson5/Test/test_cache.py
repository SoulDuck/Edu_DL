import unittest
import dog_breed
import numpy as np
from keras.applications import VGG16
import sys
sys.path.append('../')
sys.path.append('../../')
import keras_transfer


class TestKerasTransfer(unittest.TestCase):
    def setUp(self):
        # Download Cifar
        dogs_extractor = dog_breed.Dog_Extractor('./dog_breed' , 2)
        self.test_imgs = dogs_extractor.imgs
        self.test_labs = dogs_extractor.labs

        print('images shape : {}'.format(np.shape(self.test_imgs)))
        print('labels shape : {}'.format(np.shape(self.test_labs)))

        self.conv_base = VGG16(weights='imagenet', include_top=False)

        keras_transfer.end2end(self.conv_base ,[256,10])
    def test_end2end(self):
        pass;


    def test_cahce(self):

        # Test for Cache Method in ImageAugmenatation
        # test List
        features , labels = keras_transfer.cache(self.conv_base,  self.test_imgs, self.test_labs, batch_size=20)
        np.save('./tmp/features.npy' , features)
        np.save('./tmp/labels.npy', labels)

        pass;
    def tearDown(self):
        pass;

