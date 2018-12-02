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

        self.conv_base = VGG16(weights='imagenet', include_top=False , input_shape=(255,255,3))


    def test_end2end(self):
        # Test 에서 확인된것들
        # Convnet 은 Freezing 된 것
        keras_transfer.pretrained_end2end(self.conv_base, [256, 10])


    def test_cache(self):

        # Test for Cache Method in ImageAugmenatation
        # test List
        features , labels = keras_transfer.cache(self.conv_base,  self.test_imgs, self.test_labs, batch_size=20)
        np.save('./tmp/features.npy' , features)
        np.save('./tmp/labels.npy', labels)

        print('features shape : ', np.shape(features))
        print('labels shape : ',np.shape(labels))
        pass;

    def test_pretrained_dense_layer(self):
        features = np.load('tmp/features.npy')
        labels = np.load('tmp/labels.npy')
        model = keras_transfer.pretrained_dense_layer([1024,120])

    def test_pretrained_compile(self):
        features = np.load('tmp/features.npy')
        labels = np.load('tmp/labels.npy')
        model = keras_transfer.pretrained_dense_layer([1024,120])
        keras_transfer.pretrained_compile(model)

    def test_pretrained_train(self):
        features = np.load('tmp/features.npy')
        n,h,w,ch = np.shape(features)
        features = np.reshape(features , [-1 , h*w*ch])
        labels = np.load('tmp/labels.npy')
        model = keras_transfer.pretrained_dense_layer([1024,120])
        model = keras_transfer.pretrained_compile(model)
        keras_transfer.pretrained_train(model, features, labels, 10, 20, features, labels)

    def tearDown(self):
        pass;

