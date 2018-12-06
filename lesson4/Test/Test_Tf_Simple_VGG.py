import sys
sys.path.append('../')
sys.path.append('../../')
import unittest
import cifar
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import tf_simple_vgg

class TestTFSimpleVGG(unittest.TestCase):
    def setUp(self):
        # Cifar10 dataprovider
        cifar10 = cifar.Cifar10()
        self.train_imgs = cifar10.train_imgs
        self.train_labs = cifar10.train_imgs
        self.test_imgs = cifar10.test_imgs
        self.test_labs = cifar10.test_labs
        self.val_imgs = cifar10.val_imgs
        self.val_labs = cifar10.val_labs

        # Cast float to int
        self.val_imgs = np.asarray(self.val_imgs * 255).astype(np.uint8)

        # Resize cifar image 32 to 224
        tmp_list = []
        for i in range(len(self.val_imgs)):
            img = Image.fromarray(self.val_imgs[i]).resize((224, 224), Image.ANTIALIAS)
            img = np.asarray(img)
            tmp_list.append(img)
            self.assertListEqual(list(np.shape(img)), [224, 224, 3])

        self.val_imgs = np.asarray(tmp_list)
        self.assertListEqual(list(np.shape(self.val_imgs)), [5000, 224, 224, 3])


    def test_vgg11(self):
        """
        알렉스넷이 잘 구성되었는지 학인합니다.
        :return:
        """
        tf_simple_vgg.vgg_11((None, 224, 224, 3), n_classes=120)
        tf.reset_default_graph()
    def test_vgg13(self):
        """
        알렉스넷이 잘 구성되었는지 학인합니다.
        :return:
        """
        tf_simple_vgg.vgg_13((None, 224, 224, 3), n_classes=120)
        tf.reset_default_graph()

    def test_vgg16(self):
        """
        알렉스넷이 잘 구성되었는지 학인합니다.
        :return:
        """
        tf_simple_vgg.vgg_16((None, 224, 224, 3), n_classes=120)
        tf.reset_default_graph()
    def test_vgg119(self):
        """
        알렉스넷이 잘 구성되었는지 학인합니다.
        :return:
        """
        tf_simple_vgg.vgg_19((None, 224, 224, 3), n_classes=120)
        tf.reset_default_graph()

    def test_compile_sgd(self):
        """
        GradientDescent 잘 작동하는지 확인합니다
        :return:
        """
        tf.reset_default_graph()
        ops = tf_simple_vgg.vgg_11((None, 224, 224, 3), n_classes=120)
        tf_simple_vgg.compile('sgd', ops, learning_rate=0.01)

    def test_compile_momentum(self):
        tf.reset_default_graph()
        ops = tf_simple_vgg.vgg_11((None, 224, 224, 3), n_classes=120)
        tf_simple_vgg.compile('momentum', ops, learning_rate=0.01)

    def test_compile_rmsprop(self):
        tf.reset_default_graph()
        ops = tf_simple_vgg.vgg_11((None, 224, 224, 3), n_classes=120)
        tf_simple_vgg.compile('rmsprop', ops, learning_rate=0.01)

    def test_compile_adadelta(self):
        tf.reset_default_graph()
        ops = tf_simple_vgg.vgg_11((None, 224, 224, 3), n_classes=120)
        tf_simple_vgg.compile('adadelta', ops, learning_rate=0.01)

    def test_compile_adagrad(self):
        tf.reset_default_graph()
        ops = tf_simple_vgg.vgg_11((None, 224, 224, 3), n_classes=120)
        tf_simple_vgg.compile('adagrad', ops, learning_rate=0.01)

    def test_compile_adam(self):
        tf.reset_default_graph()
        ops = tf_simple_vgg.vgg_11((None, 224, 224, 3), n_classes=120)
        tf_simple_vgg.compile('adam', ops, learning_rate=0.01)

    def test_create_session(self):
        """
        Session 과 Variable Initalization 이 잘되는지 확인합니다.
        :return:
        """
        tf.reset_default_graph()
        ops = tf_simple_vgg.vgg_11((None, 224, 224, 3), n_classes=120)
        tf_simple_vgg.compile('adagrad', ops, learning_rate=0.01)
        tf_simple_vgg.create_session('vgg_11')

    def test_training(self):
        """
        Training 이 잘 작동하는지 확인합니다
        :return:
        """
        tf.reset_default_graph()
        ops = tf_simple_vgg.vgg_11((None, 224, 224, 3), n_classes=10)
        ops = tf_simple_vgg.compile('adagrad', ops, learning_rate=0.01)

        # Create session
        # Add train_op to ops
        sess, saver, writer = tf_simple_vgg.create_session('vgg_11')
        self.assertIs(os.path.isdir('./vgg_11_logs') , True)
        self.assertIs(os.path.isdir('./vgg_11_models'), True)
        # Training
        cost = tf_simple_vgg.training(sess, 1, self.val_imgs[:3], self.val_labs[:3], ops=ops)
        self.assertIsInstance(cost, list)

        # Reset tensorflow graph

    def test_eval(self):
        """
        :return:
        """
        tf.reset_default_graph()
        ops = tf_simple_vgg.vgg_11((None, 224, 224, 3), n_classes=10)
        ops = tf_simple_vgg.compile('adagrad', ops, learning_rate=0.01)

        # add train_op to ops
        # create session
        sess, saver, writer = tf_simple_vgg.create_session('vgg_11')

        # training
        acc, cost = tf_simple_vgg.eval(sess, self.val_imgs[:3], self.val_labs[:3], ops=ops)
        self.assertIsInstance(acc, np.float32)
        self.assertIsInstance(cost, np.float32)

    def tearDown(self):
        pass