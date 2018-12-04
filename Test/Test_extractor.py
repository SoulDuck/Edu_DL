import unittest
import sys
sys.path.append('../')
from extract import DogExtractor

class TestDogExtractor(unittest.TestCase):
    def setUp(self):
        pass;
    def test_data_download(self):
        """
        [O] Download Check
        """
        dex = DogExtractor('./dog_breed')

    def test_read_data(self):
        """
        [O] 데이터를 잘 불러오는지 확인합니다
        """
        dex = DogExtractor('./dog_breed')
        image , label =dex[0]
        print('Image shape : ',image.shape)
        print('Label :', label)

    def tearDown(self):
        pass;

