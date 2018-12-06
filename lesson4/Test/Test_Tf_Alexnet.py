import sys
sys.path.append('../')
sys.path.append('../../')
import unittest
import cifar
import tensorflow as tf
import tf_alexnet
from PIL import Image
import numpy as np


class TestTFAlexnet(unittest.TestCase):
    def setUp(self):
        # Cifar10 dataprovider
        cifar10 = cifar.Cifar10()
        self.train_imgs = cifar10.train_imgs
        self.train_labs = cifar10.train_imgs
        self.test_imgss = cifar10.test_imgs
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

        # Change list to Numpy
        self.val_imgs = np.asarray(tmp_list)
        self.assertListEqual(list(np.shape(self.val_imgs)), [5000, 224, 224, 3])

    def test_generate_filter(self):
        """
        원하는 형태의 filter(or kernel) 이 생성이 되는지 확인합니다

        :return:
        """
        kerenl_shape = [3, 3, 96, 96]
        test_kernel, test_bias = tf_alexnet.generate_filter([3,3,96,96])

        # kernel check
        output = list(map(int, test_kernel.get_shape()))
        self.assertListEqual(output, kerenl_shape)

        # bias check
        output = int(test_bias.get_shape()[-1])
        self.assertIs(output, kerenl_shape[-1])
        tf.reset_default_graph()

    def test_generate_units(self):
        """
        원하는 형태의 filter(or kernel) 이 생성이 되는지 확인합니다

        :return:
        """
        n_in_units = 1000
        n_out_units = 2000
        test_units, test_bias = tf_alexnet.generate_units(n_in_units, n_out_units)

        # units check
        output = list(map(int, test_units.get_shape()))
        self.assertListEqual(output, [n_in_units, n_out_units])

        # bias check
        output = int(test_bias.get_shape()[-1])
        self.assertIs(output, int(n_out_units))
        tf.reset_default_graph()

    def test_conv(self):
        pass

    def test_fc(self):
        pass

    def test_alexnet(self):
        """
        알렉스넷이 잘 구성되었는지 학인합니다.
        :return:
        """
        tf_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf.reset_default_graph()

    def test_compile_sgd(self):
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_alexnet.compile('sgd', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_compile_momentum(self):
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_alexnet.compile('momentum', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_compile_rmsprop(self):
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_alexnet.compile('rmsprop', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_compile_adadelta(self):
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_alexnet.compile('adadelta', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_compile_adagrad(self):
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_alexnet.compile('adagrad', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_compile_adam(self):
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_alexnet.compile('adam', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_create_session(self):
        """
        Session 과 Variable Initalization 이 잘되는지 확인합니다.
        :return:
        """
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_alexnet.compile('adagrad', ops, learning_rate=0.01)
        tf_alexnet.create_session('alexnet')
        tf.reset_default_graph()

    def test_training(self):
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=10)
        ops = tf_alexnet.compile('adagrad', ops, learning_rate=0.01)

        # Create session
        # Add train_op to ops
        sess, saver= tf_alexnet.create_session('alexnet')
        train_writer = tf.summary.FileWriter(logdir='./alexnet_logs/train')
        train_writer.add_graph(tf.get_default_graph())


        # Training
        g_step = tf_alexnet.training(sess, self.val_imgs[:60], self.val_labs[:60], ops, train_writer, 1, 10)
        g_step = tf_alexnet.training(sess, self.val_imgs[:60], self.val_labs[:60], ops, train_writer, g_step, 10)
        # Reset tensorflow graph
        tf.reset_default_graph()

    def test_eval(self):
        """
        Evaluation 하는 코드를 검증하니다

        :return:
        """
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=10)
        ops = tf_alexnet.compile('adagrad', ops, learning_rate=0.01)

        # add train_op to ops
        # create session
        sess, saver = tf_alexnet.create_session('alexnet')
        test_writer = tf.summary.FileWriter(logdir='./alexnet_logs/test')
        test_writer.add_graph(tf.get_default_graph())

        # training
        tf_alexnet.eval(sess, self.val_imgs[:60], self.val_labs[:60], ops=ops, writer= test_writer, global_step=0)
        tf_alexnet.eval(sess, self.val_imgs[:60], self.val_labs[:60], ops=ops, writer=test_writer, global_step=100)
        tf_alexnet.eval(sess, self.val_imgs[:60], self.val_labs[:60], ops=ops, writer=test_writer, global_step=200)
        #
        tf.reset_default_graph()

    def tearDown(self):
        pass;