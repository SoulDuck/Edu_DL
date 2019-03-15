from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
import cv2
import numpy as np


class ImageTransformer(BaseEstimator, TransformerMixin):
    """
    Base Image Transformer
    scikit-learn의 BaseEstimator를 상속받아 구현된 것으로,

    ImageTransformer.apply를 구현하면,
    image의 shape가
        - (batch_size, height, width, channel)
        - (height, width, channel)
    에 무관하게 작동.

    ImageTransformer.preprocess는 One Image에 적용하는
    Image Preprocessing 코드를 구현하면 됨

    """
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, images):
        if isinstance(images, list) or isinstance(images, tuple):
            return np.array([self.apply(image) for image in images])
        elif isinstance(images, np.ndarray):
            if images.ndim == 4:
                return np.array([self.apply(image) for image in images])
            elif images.ndim == 3:
                return self.apply(images)

    def apply(self, image):
        raise NotImplementedError("apply를 구현해야합니다.")

    def __call__(self, images):
        return self.transform(images)


class Normalization(ImageTransformer):
    """
    이미지를 0~1범위로 정규화

    example
    >>> image = np.ones((224,224,3),dtype=np.uint8)
    >>> normalization = Normalization()
    >>> result_image = normalization(image)
    >>> result_image = normalization.transform(image)

    """
    def apply(self, image):
        blank = np.zeros_like(image)
        return cv2.normalize(image, blank, 0., 1., cv2.NORM_MINMAX, cv2.CV_32F)


class RandomFlip(ImageTransformer):
    """
    이미지를 50% 확률로 좌우 반전

    example
    >>> image = np.ones((224,224,3),dtype=np.uint8)
    >>> randomflip = RandomFlip()
    >>> result_image = randomflip(image)
    >>> result_image = randomflip.transform(image)
    """
    def apply(self, image):
        if np.random.random() > 0.5:
            return image[:, ::-1, :]
        else:
            return image


class RandomRescaleAndCrop(ImageTransformer):
    """
    이미지를 [1,max_ratio] 사이의 값으로 rescale 후, random crop

    example
    >>> image = np.ones((224,224,3),dtype=np.uint8)
    >>> randomrescale = RandomRescaleAndCrop(max_ratio=1.3)
    >>> result_image = randomrescale(image)
    >>> result_image = randomrescale.transform(image)
    """
    def __init__(self, max_ratio):
        super().__init__()
        self.max_ratio = max_ratio

    def apply(self, image):
        height, width = image.shape[:2]
        rescale_ratio = np.random.uniform(1., self.max_ratio)
        image = cv2.resize(image, None, fx=rescale_ratio, fy=rescale_ratio)
        scaled_height, scaled_width = image.shape[:2]
        try:
            crop_y = np.random.randint(0, scaled_height - height)
        except BaseException:
            # scaled_height == height일 때 실행
            crop_y = 0
        try:
            crop_x = np.random.randint(0, scaled_width - width)
        except BaseException:
            # scaled_width == width일 때 실행
            crop_x = 0

        return image[crop_y:crop_y + height, crop_x:crop_x + width]


class RandomRotation(ImageTransformer):
    """
    이미지를 [-max_degree,max_degree] 사이의 값으로 회전

    example

    >>> image = np.ones((224,224,3),dtype=np.uint8)
    >>> randomrotation = RandomRotation(max_degree=20)
    >>> result_image = randomrotation(image)
    >>> result_image = randomrotation.transform(image)

    """

    def __init__(self, max_degree=30):
        super().__init__()
        """
        :param max_degree: rotation 할 때 최대 회전 각도(degree, 도), [-max_degree,max_degree] 사이의 값으로 회전함
        """
        if max_degree < 0:
            raise ValueError("max_degree는 0보다 큰 수이어야 합니다.")
        self.max_degree = int(max_degree)

    def apply(self, image):
        height, width = image.shape[:2]
        degree = np.random.randint(-self.max_degree, self.max_degree)
        M = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
        return cv2.warpAffine(image, M, (width, height))


class RandomColorShift(ImageTransformer):
    """
    이미지를 [-max_shift,max_shift] 사이의 값으로 값을 이동

    example

    >>> image = np.ones((224,224,3),dtype=np.uint8)
    >>> randomshift = RandomColorShift(max_shift=20)
    >>> result_image = randomshift(image)
    >>> result_image = randomshift.transform(image)

    """

    def __init__(self, max_shift=10):
        super().__init__()
        if max_shift < 0:
            raise ValueError("max_shift는 0보다 큰 수이어야 합니다.")
        self.max_shift = int(max_shift)

    def apply(self, image):
        shift_matrix = np.zeros_like(image, dtype=np.int)
        for i in range(3):
            shift_matrix[...,
                         i] += np.random.randint(-self.max_shift,
                                                 self.max_shift)
        image = np.clip(image + shift_matrix, 0, 255)
        return image.astype(np.uint8)
