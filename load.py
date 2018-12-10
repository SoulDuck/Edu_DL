import numpy as np


class DogDataGenerator:
    """
    StanFord Dog Breed DataGenerator
    reference : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, extractor, extractor_index=None, pipeline=None,
                 onehot=True, batch_size=32, shuffle=True):
        """

        :param extractor: extractor.DogExtractor
        :param extractor_index: extractor에서 가져올 해당 index, None이면, 전체 index
        :param pipeline: 적용할 데이터 전처리 파이프라인 (transform.py에 구현된 파이프라인)
        :param onehot: One-Hot Vector로 출력할 것인지, code로 출력할 것인지 결정
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
        self.onehot = onehot
        self.batch_size = batch_size

        self.counter = -1
        self.num_steps = int(
            np.floor(len(self.extractor_indexes) / self.batch_size))

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        :return: the number of steps ( 전체 데이터셋 수 // 배치 크기 )
        """
        return self.num_steps

    def __getitem__(self, index):
        """
        해당 step index에서의 batch images and onehots

        :param index: the index of steps in data
        :return:
        """
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]
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

    def next_batch(self):
        """
        batch size 을 주면 random으로 해당 batch size 만큼의 images, labels을 넘겨 줍니다.

        :return:
        """
        if self.counter >= self.num_steps:
            self.on_epoch_end()
            self.counter = 0
        else:
            self.counter += 1

        indexes = self.indexes[self.counter * self.batch_size:
                               (self.counter + 1) * self.batch_size]
        batch_xs, batch_ys = self.extractor[indexes]
        batch_xs = self.pipeline.transform(batch_xs)

        if self.onehot:
            batch_ys = self.extractor.convert_to_onehot(batch_ys)
        else:
            batch_ys = self.extractor.convert_to_code(batch_ys)

        return batch_xs, batch_ys
