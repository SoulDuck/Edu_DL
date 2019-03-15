import numpy as np
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


# Conv Feature Extractor
def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('{}_summaries'.format(name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def ops_summaries(ops):
    tf.summary.scalar('cost', ops['cost_op'])
    tf.summary.scalar('accuracy', ops['acc_op'])
    """
    if you want add image to tensorboard, uncomment this line 
    tf.summary.image('input', ops['x'], 16)
    """

def plot_images(images, labels=None, save_path=None):
    if isinstance(images, np.ndarray) and images.ndim == 3:
        images = [images]
        labels = [labels]
    h = np.ceil(np.sqrt(len(images)))
    fig = plt.figure(figsize=(h*6,h*6))
    plt.axis('off')
    for ind in range(len(images)):
        ax = fig.add_subplot(h, h, ind + 1)
        ax.imshow(images[ind])
        if labels is not None:
            ax.set_title(labels[ind])
    if not save_path is None:
        plt.savefig(save_path)
    plt.show()
    plt.close()

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

