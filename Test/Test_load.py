import unittest
import sys
sys.path.append('../')
from load import DogDataGenerator
import extract
class TestLoad(unittest.TestCase):
    def setUp(self):
        self.dex = extract.DogExtractor('./dog_breed')
    def test_init(self):
        dex_loader = DogDataGenerator(self.dex)
        dex_loader[0]

    def test_len(self):
        pass;
    def tearDown(self):
        pass;

