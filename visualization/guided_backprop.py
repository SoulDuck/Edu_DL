import sys
sys.path.append('../')
import keras
import tensorflow
import os
from download import download_data_url


# VGG 16 S3 download
VGG_16_url = 'https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl'
dirpath = './vgg_16'
download_data_url(VGG_16_url , dirpath)


# Unpickle
try:
    import cPickle as pickle
except ImportError:
    # Python 3
    import pickle
    model_path = os.path.join(dirpath , 'vgg16.pkl')
    with open(model_path , 'rb') as f:
        model = pickle.load(f, encoding='latin-1')
else:
    # Python 2
    with open('vgg16.pkl', 'rb') as f:
        model = pickle.load(f)

weights = model['param values']  # list of network weight tensors
classes = model['synset words']  # list of class names
mean_pixel = model['mean value']  # mean pixel value (in BGR)
del model