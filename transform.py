from sklearn.base import TransformerMixin, BaseEstimator
import cv2
import numpy as np


class Normalization(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, images):
        if images.ndim == 4:
            return np.array([self._transform(image) for image in images])
        elif images.ndim == 3:
            return self._transform(images)

    def _transform(self, image):
        blank = np.zeros_like(image)
        return cv2.normalize(image, blank, 0., 1., cv2.NORM_MINMAX, cv2.CV_32F)

    def __call__(self, images):
        return self.transform(images)


class RandomFlip(BaseEstimator, TransformerMixin):
    def __int__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, images):
        if images.ndim == 4:
            return np.array([self._transform(image) for image in images])
        elif images.ndim == 3:
            return self._transform(images)

    def _transform(self, image):
        if np.random.random() > 0.5:
            return image[:,::-1,:]
        else:
            return image

    def __call__(self, images):
        return self.transform(images)


class RandomRescaleAndCrop(BaseEstimator, TransformerMixin):
    def __init__(self, max_ratio=1.2):
        if max_ratio < 1.0:
            raise ValueError("max_ratio는 1보다 큰 수여야 합니다.")
        self.max_ratio = 1.2

    def fit(self, X, y=None):
        return self

    def transform(self, images):
        if images.ndim == 4:
            return np.array([self._transform(image) for image in images])
        elif images.ndim == 3:
            return self._transform(images)

    def _transform(self, image):
        height, width = image.shape[:2]
        rescale_ratio = np.random.uniform(1., self.max_ratio)
        image = cv2.resize(image, None, fx=rescale_ratio, fy=rescale_ratio)
        scaled_height, scaled_width = image.shape[:2]
        try:
            crop_y = np.random.randint(0, scaled_height - height)
        except:
            # scaled_height == height일 때 실행
            crop_y = 0
        try:
            crop_x = np.random.randint(0, scaled_width - width)
        except:
            # scaled_width == width일 때 실행
            crop_x = 0

        return image[crop_y:crop_y + height, crop_x:crop_x + width]

    def __call__(self, images):
        return self.transform(images)



