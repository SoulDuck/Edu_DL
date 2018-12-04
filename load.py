from keras.utils import Sequence
import numpy as np


class DogDataGenerator(Sequence):
    """
    StanFord Dog Breed DataGenerator
    reference : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, extractor, extractor_index=None, pipeline=None,
                 batch_size=32, shuffle=True):
        """

        :param extractor: extractor.DogExtractor
        :param extractor_index: extractor에서 가져올 해당 index, None이면, 전체 index
        :param pipeline: 적용할 데이터 전처리 파이프라인 (transform.py에 구현된 파이프라인)
        :param batch_size: 배치 크기
        :param shuffle: 순서를 섞을 것인지 유무
        """
        self.extractor = extractor

        if extractor_index is None:
            self.extractor_indexes = self.extractor.index
        else:
            self.extractor_indexes = extractor_index
        self.indexes = np.arange(len(self.extractor_indexes))

        self.pipeline = pipeline
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        :return: the number of steps ( 전체 데이터셋 수 // 배치 크기 )
        """
        return int(np.floor(len(self.extractor_indexes) / self.batch_size))

    def __getitem__(self, index):
        """
        해당 step index에서의 batch images and onehots

        :param index: the index of steps in data
        :return:
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_indexes = [self.extractor_indexes[k] for k in indexes]

        # Generate data
        images, labels = self.extractor[batch_indexes]
        if self.pipeline:
            images = self.pipeline.transform(images)
        onehots = self.extractor.convert_to_onehot(labels)

        return images, onehots

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.extractor_indexes))
        if self.shuffle:
            np.random.shuffle(self.indexes)

