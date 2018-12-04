import unittest
import sys
sys.path.append('../')
from extract import DogExtractor
from load import DogDataGenerator
from keras.layers import Input
from keras_alexnet import alexnet , logits , training
import configure as cfg

class TestKerasAlexnet(unittest.TestCase):
    def setUp(self):
        dex = DogExtractor('../../data/dog_breed')
        self.doggen = DogDataGenerator(dex)

    def test_keras_alexnet(self):
        """
        [ O ] Alexnet 의 구조를 확인합니다

        """
        x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
        top_conv = alexnet(x)
        pred = logits(top_conv)
        training(x, pred , self.doggen)

    def test_check_tensorboard_graph(self):
        """
        - Tensorboard 그래프를 그리고 구조가 잘 작성 되어 있는지 확인합니다
        - Validation Loss Graph , Validation Accuracy Graph
        - Train Loss Graph , Train Accuracy Graph

        :return:
        """

    def tearDown(self):
        pass;