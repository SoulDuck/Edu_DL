import unittest
import sys
import numpy as np
sys.path.append('../')
from sklearn.pipeline import Pipeline
from load import DogDataGenerator
from transform import (Normalization, RandomFlip,
                       RandomRescaleAndCrop, RandomRotation,
                       RandomColorShift)
import extract
from sklearn.model_selection import StratifiedShuffleSplit

class TestTransform(unittest.TestCase):
    def setUp(self):
        self.dog_dir = '../data/dog_breed'
        self.dex = extract.DogExtractor(self.dog_dir)
        # Image Preprocessing
        randomflip = RandomFlip()
        randomrescale = RandomRescaleAndCrop(max_ratio=1.3)
        randomrotation = RandomRotation(max_degree=30)
        randomshift = RandomColorShift(max_shift=20)
        normalization = Normalization()  # 0~255 -> 0~1로 정규화

        self.train_pipeline = Pipeline([
            ("랜덤으로 좌우 반전", randomflip),
            ("랜덤으로 rescale 후 crop", randomrescale),
            ("랜덤으로 rotation", randomrotation),
            ("랜덤으로 RGB color의 값을 shift", randomshift),
            ('0~1 범위로 정규화', normalization)
        ])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
        self.train_index, self.test_index = next(split.split(self.dex.index, self.dex.labels))

    def test_init(self):
        gen = DogDataGenerator(self.dex, self.train_index, self.train_pipeline, batch_size=16)
    def test_len(self):
        pass;
    def tearDown(self):
        pass;

