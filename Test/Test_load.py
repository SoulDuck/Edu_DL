import unittest
import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages')
sys.path.append('../')
from load import DogDataGenerator
import extract
class TestLoad(unittest.TestCase):
    def setUp(self):
        self.dex = extract.DogExtractor('./dog_breed')
    def test_init(self):
        dex_loader = DogDataGenerator(self.dex)
    def test_len(self):
        pass;
    def tearDown(self):
        pass;

