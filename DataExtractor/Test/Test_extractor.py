import sys
import unittest

sys.path.append('../')
from DataExtractor.extract import DogExtractor
import numpy as np
import os


class TestDogExtractor(unittest.TestCase):
    def setUp(self):
        self.dex = DogExtractor('../data/dog_breed')
        pass;

    def test_data_download(self):
        """
        [O] Download Check
        [O] 다운로드를 진행했다면 다시 다운로드 하지 않기
        """

        # Download Check
        data_dir = '../dog_breed'
        dex = DogExtractor(data_dir)
        n_samples = 30
        images, labels = dex[: n_samples]
        self.assertListEqual(list(np.shape(images)), [n_samples, 224, 224, 3])
        self.assertEqual(len(labels), n_samples)

        # Imports already downloaded data
        DogExtractor(data_dir)

        # Delete data
        os.remove(data_dir)

    def test_read_data(self):
        """
        [O] 데이터를 잘 불러오는지 확인합니다

        """
        dex = DogExtractor('../data/dog_breed')
        image , label =dex[0]

    def Test_inputlist(self):
        """
        : Extractor 에 input 으로 list 을 넣었을때 작동 보장
        :return:
        """

        dex = DogExtractor('../data/dog_breed')
        indices = [1,3,5,7,9]
        images, labels = dex[indices]
        self.assertEqual(len(images) , 5)

    def tearDown(self):
        pass;