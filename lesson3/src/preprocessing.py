import tensorflow as tf
import PIL
import sys,os
import matplotlib
if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import csv
from os import listdir
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def get_cifar_data() :
    train_files = listdir("train")
    test_files = listdir("test")
    mapping_info = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, \
                        'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    labeling = {}

    with open('./train/trainLabels.csv', 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader :
            labeling[row[0]] = row[1]

    with open('./test/testLabels.csv', 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader :
            labeling[row[0]] = row[1]        

    train_imgs = []
    train_labs = []

    test_imgs = []
    test_labs = []

    for f in train_files :
        point_index = f.rfind(".")
        img_id = f[0:point_index]
        try : 
            img = Image.open("train/"+f)
        except :
            continue
        train_imgs.append(np.array(img))

        label = np.zeros((10,), dtype=np.int)
        label[mapping_info[labeling[img_id]]] = 1

        train_labs.append(label)

    for f in test_files :
        point_index = f.rfind(".")
        img_id = f[0:point_index]
        try : 
            img = Image.open("test/"+f)
        except :
            continue
        test_imgs.append(np.array(img))

        label = np.zeros((10,), dtype=np.int)
        label[mapping_info[labeling[img_id]]] = 1

        test_labs.append(label)

    train_imgs = np.array(train_imgs) / float(255)
    test_imgs = np.array(test_imgs) / float(255)

    train_labs = np.array(train_labs)
    test_labs = np.array(test_labs)
    
    return train_imgs, train_labs, test_imgs, test_labs



def img2str(img):
    debug_lv0 = False

    if type(img) == str:
        img=Image.open(img)
    if not type(img).__module__ == np.__name__:
        if __debug__ ==debug_lv0:
            print 'input image type is not numpy , type was changed to numpy'
        img=np.asarray(img)
        str_=img.reshape([-1])
        if __debug__ ==debug_lv0:
            print 'image shape:',np.shape(img)
            print 'str length :',len(str_)
        return str_



def mapping(filepath , mapping_info):

    f=open(filepath)
    lines=f.readlines()
    values=[]
    for line in lines[1:]:
        str_=line.split(',')[1]
        str_=str_.replace('\n','')
        str_=str_.replace('\r','')
        value=mapping_info[str_]
        values.append(value)
    return values

def cls2onehot(cls, depth):
    debug_flag=False
    if not type(cls).__module__ == np.__name__:
        cls=np.asarray(cls)
    cls=cls.astype(np.int32)
    debug_flag = False
    labels = np.zeros([len(cls), depth] , dtype=np.int32)
    for i, ind in enumerate(cls):
        labels[i][ind:ind + 1] = 1
    if __debug__ == debug_flag:
        print '#### data.py | cls2onehot() ####'
        print 'show sample cls and converted labels'
        print cls[:10]
        print labels[:10]
        print cls[-10:]
        print labels[-10:]
    return labels

def get_cifar(train_folder='./train' , test_folder='./test' , type_='str'):
    """
    airplane :0
    automobile :1
    bird :2
    cat :3
    deer :4
    dog :5
    frog :6
    horse :7
    ship :8
    truck:9
    """
    mapping_info = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, \
                    'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    train_paths = glob.glob(os.path.join(train_folder, '*.png'))
    test_paths = glob.glob(os.path.join(test_folder, '*.png'))
    n_train = len(train_paths)
    n_test = len(test_paths)
    train_paths.sort(key=natural_keys)
    test_paths.sort(key=natural_keys)

    print '# of train data : ', n_train
    print '# of test data : ', n_test

    if type_ == 'str':
        train_imgs=map(img2str , train_paths)
        test_imgs = map(img2str, test_paths)
    else:
        train_imgs=map(lambda path : np.asarray(Image.open(path)) , train_paths)
        test_imgs = map(lambda path: np.asarray(Image.open(path)), test_paths)

    print 'shape of train data : ', np.shape(train_imgs)
    print 'shape of test data : ', np.shape(test_imgs)
    train_cls = mapping('./train/trainLabels.csv', mapping_info)
    test_cls = mapping('./test/testLabels.csv', mapping_info)
    train_labs = cls2onehot(train_cls, depth=10)
    test_labs = cls2onehot(test_cls, depth=10)

    print 'shape of train data : ', np.shape(train_labs)
    print 'shape of test data : ', np.shape(test_labs)

    return train_imgs , train_labs , test_imgs, test_labs





if __name__ =='__main__':
    #print mapping('./train/trainLabels.csv' , mapping_info)
    train_imgs, train_labs, test_imgs, test_labs=get_cifar(type_='image')


