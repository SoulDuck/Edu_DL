import csv
import numpy as np
import pandas as pd
import math
import random
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

def readlines_csv(csv_path):
    read_f = open(csv_path, 'r')
    lines = read_f.readlines()
    for l in lines:
        print(l)


def change_element(df, row_n, col_n, text ):

    df.iloc[row_n, col_n] = text


def change_elements(csv_path, row_list, col_list, texts):
    df = pd.read_csv(csv_path)
    assert len(row_list) == len(col_list) == len(texts)

    rows_cols_texts = zip(row_list , col_list,texts)
    for r, c, t in rows_cols_texts:
        df.iloc[r, c] = t
    df.to_csv(csv_path,index=False)

def plot_images(imgs, names=None, random_order=False, savepath=None, no_axis=True):
    h = math.ceil(math.sqrt(len(imgs)))
    fig = plt.figure()
    for i in range(len(imgs)):
        ax = fig.add_subplot(h, h, i + 1)
        if random_order:
            ind = random.randint(0, len(imgs) - 1)
        else:
            ind = i
        img = imgs[ind]
        plt.axis('off')
        plt.imshow(img)
        if not names == None:
            ax.set_xlabel(names[ind])
    if not savepath is None:
        plt.savefig(savepath)
    plt.show()


def show_progress(step , max_step):
    msg = '\r {}/{}'.format(step, max_step)
    sys.stdout.write(msg)
    sys.stdout.flush()

def path2np(path , resize):
    if not resize is None:
        np_img = np.asarray(Image.open(path).convert('RGB').resize(resize))
    else:
        np_img = np.asarray(Image.open(path).convert('RGB'))

    return np_img


def paths2np(paths ,resize = None):
    ret_np=[]
    error_indices = []
    for i,p in enumerate(paths):
        show_progress(i,len(paths))
        try:
            ret_np.append(path2np(p, resize))
        except IOError as ioe:
            print('Error Path : {}'.format(p))
            error_indices.append(i)
    return np.asarray(ret_np) , error_indices

def get_extension(path):
    return os.path.splitext(os.path.split(path)[-1])[-1]



def cls2onehot(cls , depth):
    labs=np.zeros([len(cls) , depth])

    for i,c in enumerate(cls):
        labs[i,c]=1
    return labs

