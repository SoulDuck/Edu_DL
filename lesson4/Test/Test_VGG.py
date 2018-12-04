import unittest
import sys
sys.path.append('../')

from extract import DogExtractor
from load import DogDataGenerator
from keras.layers import Input
from keras_vgg import vgg11, vgg13, vgg16, vgg19, logits , training
import configure as cfg

class TestKerasAlexnet(unittest.TestCase):
    def setUp(self):
        dex = DogExtractor('../../data/dog_breed')
        self.doggen = DogDataGenerator(dex)
    def test_keras_vgg11(self):
        """
        [ O ] VGG11 의 구조를 확인합니다

        """
        x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
        top_conv = vgg11(x)
        pred = logits(top_conv, cfg.n_classes)
        training(x, pred , self.doggen)

    def test_keras_vgg13(self):
        """
        [ O ] VGG13 의 구조를 확인합니다

        """
        x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
        top_conv = vgg13(x)
        pred = logits(top_conv, cfg.n_classes)
        training(x, pred , self.doggen)

    def test_keras_vgg16(self):
        """
        [ O ] VGG16 의 구조를 확인합니다

        """
        x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
        top_conv = vgg16(x)
        pred = logits(top_conv, cfg.n_classes)
        training(x, pred , self.doggen)

    def test_keras_vgg19(self):
        """
        [ O ] VGG19 의 구조를 확인합니다

        """
        x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
        top_conv = vgg19(x)
        pred = logits(top_conv, cfg.n_classes)
        training(x, pred , self.doggen)


    def tearDown(self):
        pass;