import sys
sys.path.append('../')
sys.path.append('../../')
import unittest
import cifar
import tf_simple_alexnet
import tensorflow as tf
from PIL import Image
import numpy as np


class TestTFSimpleAlexnet(unittest.TestCase):
    def setUp(self):
        # Cifar10 dataprovider
        cifar10 = cifar.Cifar10()
        self.train_imgs = cifar10.train_imgs
        self.train_labs = cifar10.train_labs
        self.test_imgs = cifar10.test_imgs
        self.test_labs = cifar10.test_labs
        self.val_imgs = cifar10.val_imgs
        self.val_labs = cifar10.val_labs

    def test_alexnet(self):

        # Training
        tf_simple_alexnet.alexnet([None,32,32,3],10)
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        learning_rate = tf.get_default_graph().get_tensor_by_name('learning_rate:0')
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')

        # Train Filewriter
        writer = tf.summary.FileWriter('tmp_log')

        cost_op = tf.get_default_graph().get_tensor_by_name('cost_op:0')
        acc_op = tf.get_default_graph().get_tensor_by_name('acc_op:0')
        # acc_op = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)[0]
        train_op = tf.get_collection(tf.GraphKeys.TRAIN_OP)[0]
        merge_op = tf.summary.merge_all()

        #
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        fetches = [train_op, cost_op, acc_op,merge_op]
        feed_dict = {x: self.train_imgs[:60], y: self.train_labs[:60], phase_train: True, learning_rate: 0.01,
                     keep_prob: 0.5}

        for i in range(5):
            _, cost, acc, summary_merge = sess.run(fetches=fetches, feed_dict=feed_dict)
            writer.add_summary(summary_merge, global_step=i)
            print('cost : {} , acc : {}'.format(cost, acc))





