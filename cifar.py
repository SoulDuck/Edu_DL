from urllib import request
import os ,sys
import zipfile
import tarfile
import glob
import numpy as np
import pickle
import random


def cls2onehot(cls, depth):
    labs = np.zeros([len(cls), depth])
    for i, c in enumerate(cls):
        labs[i, c] = 1
    return labs


def next_batch(imgs, labs, batch_size):
    indices = random.sample(range(np.shape(imgs)[0]), batch_size)
    if not type(imgs).__module__ == np.__name__:  # check images type to numpy
        imgs = np.asarray(imgs)
    imgs = np.asarray(imgs)
    batch_xs = imgs[indices]
    batch_ys = labs[indices]
    return batch_xs, batch_ys


class Cifar10(object):
    def __init__(self):
        self.url = 'http://www.cs.toronto.edu/~kriz/cifar-%d-python.tar.gz' % 10
        self.img_size = 32
        # Number of channels in each image, 3 channels: Red, Green, Blue.
        self.num_channels = 3
        # Length of an image when flattened to a 1-dim array.
        self.img_size_flat = self.img_size * self.img_size * self.num_channels
        # Number of classes.
        self.num_classes = 10

        self.file_path = self.download_data_url(self.url, './cifar_10')
        self.unzip(self.file_path, extract_dir='./cifar_10' )
        train_filenames = glob.glob('./cifar_10/cifar-10-batches-py/data_batch*')
        test_filenames = glob.glob('./cifar_10/cifar-10-batches-py/test_batch*')
        self.train_imgs, self.train_labs = self.get_images_labels(*train_filenames)
        self.test_imgs, self.test_labs = self.get_images_labels(*test_filenames)

        self.val_imgs = self.test_imgs[5000:]
        self.test_imgs = self.test_imgs[:5000]
        self.val_labs = self.test_labs[5000:]
        self.test_labs = self.test_labs[:5000]

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

    @classmethod
    def get_images_labels(cls, *filenames):
        images = []
        labels = []
        for i, f in enumerate(filenames):
            with open(f, mode='rb') as file:
                data = pickle.load(file, encoding='latin1')
                images.append(data['data'].reshape([-1, 3, 32, 32]))
                labels.extend(data['labels'])

        images = np.vstack(images)
        images = images.transpose([0, 2, 3, 1])
        labels = cls2onehot(labels, 10)
        images = images/255.
        print('labels shape : ', np.shape(labels))
        print('imagess shape : ', np.shape(images))
        return images, labels

