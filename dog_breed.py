import os
import zipfile
import tarfile
import csv
from utils import paths2np, cls2onehot
import numpy as np
import sys
from utils import get_extension
from urllib import request
import random


class Extracter(object):
    def __init__(self):
        pass

    def generate_datainfo(self, datainfo_path, image_dir, n_limit=None):
        """
        TODO : Dataframe 을 이용하는 것으로 바꾸기
        이 함수는 폴더 구성이 아래와 같음을 예상하고 만들었음
        images
            |- dog1
                |-1.jpg
                |-2.jpg
                ...
            |- dog2
                |-1.jpg
                |-2.jpg
                ...

        filepath , labelname , label(number) 가 저장됩니다.
        Mockup 파일이 아니면 limit_count 을 특정 숫자로 지정해 주면 됩니다.

        :return:
        """
        # Generate csv file
        f = open(datainfo_path, 'w')
        writer = csv.DictWriter(f, fieldnames=["filepath", "labelname", "label" ])
        # write CSV Head
        writer.writerow({"filepath": 'filepath', "labelname": "labelname", "label": "label"})
        label_count = 0

        # Crawling target dir
        for i, (rt_dir, subdir, files) in enumerate(os.walk(image_dir)):
            if i == 0:
                # i==0 은 시작한 자기 폴더라서 뛰어넘어버립니다.
                pass;
            else:
                labelname = os.path.split(rt_dir)[-1]
                for count,file in enumerate(files[:n_limit]):
                    file_ext = get_extension(file)
                    # Checking image files
                    if file_ext in ['.jpg', 'jpeg', 'png', 'JPG', 'JPEG']:
                        filepath = os.path.join(rt_dir, file)
                        # write path , labelname , label to csv file
                        writer.writerow({"filepath": filepath, "labelname": labelname, "label": label_count})
                print('{} : {} '.format(rt_dir, label_count))
                label_count += 1
        f.close()

    def get_images_labels(self, onehot, n_classes):
        raise NotImplementedError

    @classmethod
    def get_col_elements(cls, csv_path, col_idx):
        """
        csv파일을 불러옵니다.
        특정 columne 의 모든 elements 을 가져옵니다.
        :param fpath:
        :param col_idx:
        :return:
        """
        f = open(csv_path, 'r')
        lines=f.readlines()
        ret_paths = []
        for line in lines[1:]:
            if len(line.split(',')) ==0:
                continue
            ret_paths.append(line.split(',')[col_idx])
        return ret_paths

    @classmethod
    def report_download_progress(cls, count, block_size, total_size):
        pct_complete = float(count * block_size) / total_size
        msg = "\r {0:1%} already downloaded".format(pct_complete)
        sys.stdout.write(msg)
        sys.stdout.flush()

    @classmethod
    def download_data_url(cls, url, download_dir):
        filename = url.split('/')[-1]
        file_path = os.path.join(download_dir, filename)
        if not os.path.exists(file_path):
            try:
                os.makedirs(download_dir)
            except Exception:
                pass
            print("Download %s  to %s" % (url, file_path))
            file_path, _ = request.urlretrieve(url=url, filename=file_path, reporthook=cls.report_download_progress)
        else:
            print('cifar is already downloaded')
        return file_path

    @classmethod
    def unzip(cls, file_path, extract_dir):
        print('\nExtracting files')
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extracall(extract_dir)
        elif file_path.endswith(".tar.gz"):
            tarfile.open(name=file_path, mode='r:gz').extractall(extract_dir)
        elif file_path.endswith(".tgz"):
            tarfile.open(name=file_path, mode='r:gz').extractall(extract_dir)
        elif file_path.endswith(".tar"):
            tarfile.open(name=file_path, mode='r').extractall(extract_dir)


    @classmethod
    def divide_train_val(self, np_images , labels , validation_ratio , n_classes):
        """
        비율에 따라 데이터셋을 나눕니다.
        전체 데이터 셋에 대해서 나누는 것이 아니라 class 별 데이터셋에 대해 지정된 ratio 비율로 나눕니다.

        Ex)
        validation_ratio = 0.2 , train ratio = 0.8
        n class1 = 100 | n class2 = 10

        class1 : n train = 80 | n validation = 20
        class2 : n train = 8  | n validation= 2

        :param images:
        :param labels:
        :param validation_ratio:
        :param n_classes:
        :return:
        """
        # if labels is OneHot Vector
        if np.ndim(labels)==2:
            labels = np.argmax(labels , axis=1)

        np_images = np.asarray(np_images)
        train_imgs = []
        train_labs = []
        val_imgs = []
        val_labs = []

        for class_idx in range(n_classes):
            target_indices = np.where([labels == class_idx])[1]
            n_target = len(target_indices)
            n_val = int(n_target * validation_ratio)
            n_train = int(n_target - n_val)
            print('class index : {} \t # train : {} \t # validation :{}'.format(class_idx , n_train , n_val))

            # Extract indices
            random.shuffle(target_indices)
            train_indices = target_indices[:n_train]
            val_indices = target_indices[n_train:]

            # Stack images and labels
            train_imgs.append(np_images[train_indices])
            train_labs.extend(labels[train_indices])
            val_imgs.append(np_images[val_indices])
            val_labs.extend(labels[val_indices])

        # list 2 Numpy
        train_imgs = np.vstack(train_imgs)
        val_imgs = np.vstack(val_imgs)

        return train_imgs , train_labs , val_imgs , val_labs

class Transformer(object):
    def __init__(self):
        raise NotImplementedError



class Dog_Extractor(Extracter):
    def __init__(self , image_dir , sample_per_limit = None):
        url='http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
        super(Dog_Extractor , self).__init__()
        self.sample_per_limit = sample_per_limit
        # if dataset was downloaded or the folder exists in `image_dir`
        # download is not performed
        if not os.path.isdir(image_dir):
            fpath = self.download_data_url(url, image_dir)
            # image dir : dir path to Unzip
            self.unzip(fpath , image_dir)

        self.image_dir = os.path.join(image_dir , 'Images')
        # record [filepath , label] to csv file
        self.datainfo_path = 'dataset_info.csv'

        if not self.sample_per_limit is None:
            # Mock-Up
            self.generate_datainfo(self.datainfo_path, self.image_dir, self.sample_per_limit)
            self.imgs, self.labs = self.get_images_labels(True, n_classes=120)

        else:
            # Real Data
            self.generate_datainfo(self.datainfo_path, self.image_dir)
            self.imgs, self.labs = self.get_images_labels(True, n_classes=120)
        #


    def get_images_labels(self, onehot, n_classes):
        """
        self.datainfo_path
        :param onehot:
        :param n_classes:
        :return:

        이렇게 코드를 짜면 문제가 있다.
        image 에 문제가 있을면 label에는 반영되지 않는다.
        """
        paths = self .get_col_elements(self.datainfo_path, 0)
        labels = self.get_col_elements(self.datainfo_path, 2)
        images , err_indices = paths2np(paths, (255, 255))

        # 에러가 난 index 을 labels 이 담겨있는 list에도 빼주어야 한다.
        # list 에서 element 을 추출하면 pop 을 하는동안 list 형태가 바뀌어서 거꾸로 빼줘야 한다.
        #labels = map(labels.pop , err_indices)
        err_indices.reverse()
        for idx in err_indices:
            labels.pop(idx)
        # str ==> int
        labels = list(map(int , labels))
        if onehot:
            labels = cls2onehot(labels, n_classes)
        return np.asarray(images) , np.asarray(labels)
















