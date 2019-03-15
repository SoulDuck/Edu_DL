import sys
import unittest

sys.path.append('../')
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages')
import numpy as np
from DataExtractor.load import DogDataGenerator
from DataExtractor import extract
from sklearn.pipeline import Pipeline
from DataExtractor.transform import (RandomFlip,
                                     RandomRescaleAndCrop, RandomRotation,
                                     RandomColorShift)
from sklearn.model_selection import StratifiedShuffleSplit


class TestLoad(unittest.TestCase):
    def setUp(self):

        # Extractor
        self.dex = extract.DogExtractor('/Users/seongjungkim/PycharmProjects/Edu_DL/data/mockup')
        self.batch_size = 10

        # Pipeline
        # 0.5확률로 좌우 반전
        randomflip = RandomFlip()
        # random하게 rescale 후 원래 이미지 사이즈만큼 crop
        randomrescale = RandomRescaleAndCrop(max_ratio=1.3)
        # random하게 이미지를 돌림
        randomrotation = RandomRotation(max_degree=30)
        # Randomg하게 RGB별로 값을 shift함
        randomshift = RandomColorShift(max_shift=20)
        # Normalization
        #normalization = Normalization()  # 0~255 -> 0~1로 정규화
        train_pipeline = Pipeline([
            ("랜덤으로 좌우 반전", randomflip),
            ("랜덤으로 rescale 후 crop", randomrescale),
            ("랜덤으로 rotation", randomrotation),
            ("랜덤으로 RGB color의 값을 shift", randomshift),
            #('0~1 범위로 정규화', normalization)
        ])
        images, labels = self.dex[:self.batch_size]
        images = train_pipeline.transform(images)

        # Split Train and Validation
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
        self.train_index, self.test_index = next(split.split(self.dex.index, self.dex.labels))
        self.assertEqual(len(self.dex.index), 20)
        self.assertEqual(len(self.train_index) , 14)
        self.assertEqual(len(self.test_index), 6)

        # Train index 하고 test index 하고 겹치지 않는걸 보증합니다.
        self.assertEqual(len(set(list(self.train_index)+list(self.test_index))) , 20)

        # Loader
        self.doggen = DogDataGenerator(self.dex, extractor_index=self.train_index, pipeline=train_pipeline,
                                       batch_size=self.batch_size)
        self.assertListEqual([self.batch_size, 224, 224, 3], list(np.shape(images)))

    def test_random_next_batch(self):
        """
        random 으로 batch_size 만큼의 이미지와 라벨을 뽑아 출력함
        :return:
        """
        from DataExtractor.utils import plot_images
        batch_xs , batch_ys = self.doggen.random_next_batch()
        self.assertEqual(len(batch_xs) , self.batch_size)
        self.assertEqual(len(batch_ys), self.batch_size)

        self.train_index = list(self.train_index)
        self.test_index = list(self.test_index)

        # load All Train imges
        self.train_imgs, self.train_labs =  self.dex[self.train_index]
        self.test_imgs, self.test_labs = self.dex[self.test_index]

        total_imgs = np.vstack([self.train_imgs, batch_xs])
        plot_images(total_imgs)

        # Batch xs 의 이미지가 test imgs 안에 들어 있으면 안됌
        total_imgs = np.vstack([self.test_imgs, batch_xs])
        plot_images(total_imgs)

    def test_len(self):
        pass;

    def tearDown(self):
        pass;

