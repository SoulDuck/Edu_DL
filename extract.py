import os
import tarfile
import pandas as pd
import numpy as np
import sys
from urllib import request
from tqdm import tqdm
import glob
import cv2
import warnings
from functools import lru_cache


def download_dog_bread_dataset(download_dir):
    """
    stanford에서 제공하는 dog bread classification dataset을 다운

    :param download_dir: 저장할 디렉토리
    :return:
    """
    DOG_BREAD_DATASET_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    # 데이터 받아오기(download & unzip)
    zip_path = download_data_url(DOG_BREAD_DATASET_URL, download_dir)
    image_dir = os.path.join(download_dir, "Images/")
    if not os.path.exists(image_dir):
        print("Extract Zip Folder to {}".format(image_dir))
        with tarfile.open(name=zip_path, mode='r') as zipfile:
            zipfile.extractall(download_dir)
    else:
        print("Already extracted")

    # summarize data and write datainfo
    info_path = os.path.join(download_dir, "datainfo.txt")
    if not os.path.exists(info_path):
        print("Summarize DataSet")
        info_df = summarize_directory(image_dir)
        info_df.to_csv(info_path,index=False)
    else:
        print("Already Summarized")


def download_data_url(url, download_dir):
    """
    url를 통해 데이터를 download_dir에 저장

    :param url: 가져올 주소
    :param download_dir: 저장할 디렉토리
    :return:
    """
    filename = os.path.split(url)[1]
    file_path = os.path.join(download_dir, filename)
    if not os.path.exists(file_path):
        print("Download {} to {}".format(url, file_path))
        os.makedirs(download_dir, exist_ok=True)

        def download_reporthook(count, block_size, total_size):
            """
            다운로드 진행 경과를 report하는 hook
            """
            progresses = float(count * block_size) / total_size
            msg = "\r {:2.3f}% already downloaded".format(progresses * 100)
            sys.stdout.write(msg)
            sys.stdout.flush()

        file_path, _ = request.urlretrieve(url=url, filename=file_path,
                                           reporthook=download_reporthook)
    else:
        print('data is already downloaded')
    return file_path


def summarize_directory(dataset_dir):
    """
    데이터 셋이 저장된 디렉토리의 이미지와 라벨을 summarize한 dataframe을 retrun
    dataset_dir
        |- label_1
            |- image1
            |- image2
        |- label_2
            |- image3
            |- image4
        |- label_3
            |- image5

    output : pd.DataFrame
    | file_path | label |
    | -----     | ----  |
    | dataset_dir/label_1/image1 | label_1 |
    | dataset_dir/label_1/image2 | label_1 |
    | dataset_dir/label_2/image3 | label_2 |
    | dataset_dir/label_2/image4 | label_2 |
    | dataset_dir/label_3/image5 | label_3 |

    :param dataset_dir: 저장된 데이터셋의 디렉토리(데이터셋의 root directory)
    :return:
    """
    info_df = pd.DataFrame(columns=['file_path', 'label'])
    for file_path in tqdm(glob.glob(os.path.join(dataset_dir, "*/*"))):
        label_name = file_path.split(os.sep)[-2]
        absolute_path = os.path.abspath(file_path)
        info_df = info_df.append({
            "file_path" : absolute_path,
            "label" : label_name
        },ignore_index=True)
    return info_df


class DogExtractor(object):
    """
    스탠포드 데이터셋 dog bread classification의 데이터를 extract하는 메소드

    Example
    >>> dex = DogExtractor("./data")
    >>> image, label = dex[0] # indexing by integer
    >>> images, labels = dex[0:10] # indexing by range
    >>> images, labels = dex[[1,3,5,9]] # indexing by list
    >>> dex.index[:10]
    Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    >>> dex.labels[0:1000:200]
    0                n02097658-silky_terrier
    200         n02092002-Scottish_deerhound
    400         n02092002-Scottish_deerhound
    600               n02091244-Ibizan_hound
    800    n02095314-wire-haired_fox_terrier
    Name: label, dtype: object

    """
    def __init__(self, data_dir, image_shape=(224, 224, 3), keep_aspect_ratio=True):
        """

        :param data_dir: dog bread classification이 담겨 있는 directory, 없으면 이 경로로 다운받음
        :param image_shape: 출력받을 이미지의 크기
        :param keep_aspect_ratio: 이미지의 비율을 고정시키는가 유무 (True이면, padding을 넣어서 image_shape로 resize함)

        """
        self.image_shape = image_shape
        self.keep_aspect_ratio = keep_aspect_ratio

        info_path = os.path.join(data_dir, "datainfo.txt")
        if not os.path.exists(info_path):
            download_dog_bread_dataset(data_dir)
        self.info_df = pd.read_csv(info_path)
        self.info_df.label = self.info_df.label.astype('category')
        self.n_classes = len(self.info_df.label.cat.categories)
        self.n_samples = len(self.info_df)


        def code2onehot(code):
            onehot = np.zeros(self.n_classes)
            onehot[code] = 1.
            return onehot

        self.label2onehot = {label_name : code2onehot(code)
                             for code, label_name in enumerate(self.info_df.label.cat.categories)}

    def __len__(self):
        return len(self.info_df)

    @property
    def index(self):
        return self.info_df.index

    @property
    def labels(self):
        return self.info_df.label

    def __getitem__(self, slc):
        if isinstance(slc, int):
            return self._view_by_index(slc)
        elif isinstance(slc, slice):
            # slice 보정 (pandas의 경우 slice의 stop까지 포함시킴)
            if slc.step is None or slc.step > 0:
                slc = slice(slc.start, slc.stop-1, slc.step)
            else:
                slc = slice(slc.start, slc.stop+1, slc.step)
            return self._view_by_dataframe(self.info_df.loc[slc])
        elif isinstance(slc, list):
            return self._view_by_dataframe(self.info_df.loc[slc])
        elif isinstance(slc, tuple):
            return self._view_by_dataframe(self.info_df.loc[list(slc)])
        else:
            raise ValueError("적절치 못한 indexing입니다.")

    @property
    def label_counts(self):
        return self.info_df.label.value_counts()

    @lru_cache(maxsize=30000)
    def _view_by_index(self, index):
        row = self.info_df.loc[index]
        image = self._read_image(row.file_path)
        label = row.label
        return image, label

    def _view_by_dataframe(self, view_df):
        images = []
        labels = []
        for index, _ in view_df.iterrows():
            image, label = self._view_by_index(index)
            images.append(image)
            labels.append(label)
        return np.stack(images), labels

    def _read_image(self, file_path):
        image = cv2.imread(file_path)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            warnings.warn("{} 파일을 읽을 수 없습니다.".format(file_path))
            return np.zeros(self.image_shape)

        if self.keep_aspect_ratio:
            dst_h, dst_w = self.image_shape[:2]
            src_h, src_w = image.shape[:2]
            h_ratio, w_ratio = dst_h / src_h, dst_w / src_w
            if h_ratio < w_ratio:
                image = cv2.resize(image, None, fx=h_ratio, fy=h_ratio)
                delta = self.image_shape[1] - image.shape[1]
                image = cv2.copyMakeBorder(image, 0, 0, delta // 2, delta - delta // 2,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                image = cv2.resize(image, None, fx=h_ratio, fy=w_ratio)
                delta = self.image_shape[0] - image.shape[0]
                image = cv2.copyMakeBorder(image, delta // 2, delta - delta // 2, 0, 0,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
        dst_h, dst_w = self.image_shape[:2]
        return cv2.resize(image, (dst_w, dst_h))

    def convert_to_onehot(self, labels):
        if isinstance(labels, str):
            return self.label2onehot[labels]
        else:
            return np.stack([self.label2onehot[label] for label in labels])
