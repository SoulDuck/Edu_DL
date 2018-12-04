import unittest
import sys
sys.path.append('../')
import numpy as np
from load import DogDataGenerator
import extract
class TestLoad(unittest.TestCase):
    def setUp(self):
        self.dex = extract.DogExtractor('../data/dog_breed')
    def test_init(self):
        dex_loader = DogDataGenerator(self.dex)
        images , labels = dex_loader[0]
        print(np.shape(images))
        print(np.max(images))

    def test_len(self):
        pass;
    def tearDown(self):
        pass;

