import unittest
import sys
sys.path.append('../')
from extract import DogExtractor
from load import DogDataGenerator


class TestKerasAlexnet(unittest.TestCase):
    def setUp(self):
        dex = DogExtractor('../data/dog_breed')
        self.doggen = DogDataGenerator(dex)

    def test_keras_alexnet(self):
        """
        [ O ] Alexnet 의 구조를 확인합니다

        :return:
        """
        pass;

    def tearDown(self):
        pass;