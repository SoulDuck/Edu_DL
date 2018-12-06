import sys
sys.path.append('../')
sys.path.append('../../')
import unittest
import cifar
import tensorflow as tf
import tf_simple_resnet
from PIL import Image
import numpy as np

class TestTFSimpleResnet(unittest.TestCase):
    def setUp(self):
        # Cifar10 dataprovider
        cifar10 = cifar.Cifar10()
        self.train_imgs = cifar10.train_imgs
        self.train_labs = cifar10.train_labs
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

        # kernel
        tf.reset_default_graph()
        self.kernel = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
        self.kernel[:, 3, 0, 0] = 1
        self.kernel[3, :, 0, 1] = 1

        # Input Image
        self.sample_imgs = self.val_imgs[0:1]
        self.assertIs(np.ndim(self.sample_imgs), 4)
        self.sample_imgs_node = tf.Variable(self.sample_imgs, dtype=tf.float32)

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='x')
        self.phase_train = tf.placeholder(dtype=tf.bool)

    def test_stem(self):
        """
        Convolution filter 가 잘 구성되어 있는지 확인합니다
        :return:
        """
        # Residual Block
        # Image size was keep

        layer = tf_simple_resnet.stem(self.x, self.phase_train)

        # Checking
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        output = sess.run(layer, {self.x: self.sample_imgs, self.phase_train: True})
        sess.close()
        tf.reset_default_graph()

        self.assertListEqual(list(np.shape(output)), [1, 55, 55, 64])

    def test_residual_block(self):
        """
        Convolution filter 가 잘 구성되어 있는지 확인합니다
        :return:
        """

        # Residual Block
        # Image size was keep
        layer = tf_simple_resnet.stem(self.x, self.phase_train)
        layer = tf_simple_resnet.residual_block(layer, 64, self.phase_train)

        # Checking
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        output = sess.run(layer, {self.x: self.sample_imgs, self.phase_train: True})
        sess.close()
        tf.reset_default_graph()

        self.assertListEqual(list(np.shape(output)), [1, 55, 55, 64])

    def test_residual_projection(self):
        """
        Convolution filter 가 잘 구성되어 있는지 확인합니다
        :return:
        """
        # Residual Block
        # Image size was keep
        layer = tf_simple_resnet.stem(self.x, self.phase_train)
        layer = tf_simple_resnet.residual_block_projection(layer, 64, self.phase_train)

        # Checking
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        output = sess.run(layer, {self.x: self.sample_imgs, self.phase_train: True})
        sess.close()
        tf.reset_default_graph()

        self.assertListEqual(list(np.shape(output)), [1, 28, 28, 64])

    def test_bottlenct_block(self):
        """
        Convolution filter 가 잘 구성되어 있는지 확인합니다
        :return:
        """
        # Residual Block
        # Image size is preserved
        layer = tf_simple_resnet.stem(self.x, self.phase_train)
        layer = tf_simple_resnet.bottlenect_block(layer, 64, self.phase_train)

        # Checking
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        output = sess.run(layer, {self.x: self.sample_imgs, self.phase_train: True})
        sess.close()
        tf.reset_default_graph()

        self.assertListEqual(list(np.shape(output)), [1, 55, 5, 64])

    def test_bottlenct_block_projection(self):
        """
        Convolution filter 가 잘 구성되어 있는지 확인합니다
        :return:
        """
        # Residual Block
        # Image size is reduced
        layer = tf_simple_resnet.stem(self.x, self.phase_train)
        layer = tf_simple_resnet.residual_block_projection(layer, 64, self.phase_train)

        # Checking
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        output = sess.run(layer, {self.x: self.sample_imgs, self.phase_train: True})
        sess.close()
        tf.reset_default_graph()

        self.assertListEqual(list(np.shape(output)), [1, 28, 28, 64])

    def test_resnet_create_session(self):
        layer = tf_simple_resnet.stem(self.x, self.phase_train)
        tf_simple_resnet.residual_block_projection(layer, 64, self.phase_train)
        tf_simple_resnet.create_session()

    def test_resnet_18(self):
        """
        resnet 18 graph가 잘 그려지는지 확인합니다
        :return:
        """
        tf.reset_default_graph()
        tf_simple_resnet.resnet_18([None, 224, 224, 3], 120)

    def test_resnet_34(self):
        """
        resnet 34 graph가 잘 그려지는지 확인합니다
        :return:
        """
        tf.reset_default_graph()
        tf_simple_resnet.resnet_18([None, 224, 224, 3], 120)

    def test_resnet_50(self):
        """
        resnet 50 graph가 잘 그려지는지 확인합니다
        :return:
        """
        tf.reset_default_graph()
        tf_simple_resnet.resnet_18([None, 224, 224, 3], 120)

    def test_resnet_101(self):
        """
        resnet 101 graph가 잘 그려지는지 확인합니다
        :return:
        """
        tf.reset_default_graph()
        tf_simple_resnet.resnet_18([None, 224, 224, 3], 120)

    def test_resnet_151(self):
        """
        resnet 151 graph가 잘 그려지는지 확인합니다
        :return:
        """
        tf.reset_default_graph()
        tf_simple_resnet.resnet_18([None, 224, 224, 3], 120)

    def test_compile_18(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_18([None, 224, 224, 3], 120)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        tf_simple_resnet.create_session()

    def test_training_18(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_18([None, 224, 224, 3], 10)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        sess = tf_simple_resnet.create_session()
        tf_simple_resnet.training(sess, 10, self.val_imgs, self.val_labs, 30, ops)

    def test_testing_18(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_18([None, 224, 224, 3], 10)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        sess = tf_simple_resnet.create_session()
        tf_simple_resnet.eval(sess, self.val_imgs[:30], self.val_labs[:30], 30, ops)
        pass;


    def test_compile_34(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 120)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        tf_simple_resnet.create_session()

    def test_training_34(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 10)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        sess = tf_simple_resnet.create_session()
        tf_simple_resnet.training(sess, 5, self.val_imgs, self.val_labs, 30, ops)

    def test_testing_34(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 10)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        sess = tf_simple_resnet.create_session()
        tf_simple_resnet.eval(sess, self.val_imgs[:30], self.val_labs[:30], 30, ops)
        pass;

    def test_compile_50(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 120)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        tf_simple_resnet.create_session()

    def test_training_50(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 10)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        sess = tf_simple_resnet.create_session()
        tf_simple_resnet.training(sess, 5, self.val_imgs, self.val_labs, 30, ops)

    def test_testing_50(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 10)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        sess = tf_simple_resnet.create_session()
        tf_simple_resnet.eval(sess, self.val_imgs[:30], self.val_labs[:30], 30, ops)
        pass;

    def test_compile_101(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 120)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        tf_simple_resnet.create_session()

    def test_training_101(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 10)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        sess = tf_simple_resnet.create_session()
        tf_simple_resnet.training(sess, 5, self.val_imgs, self.val_labs, 30, ops)

    def test_testing_101(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 10)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        sess = tf_simple_resnet.create_session()
        tf_simple_resnet.eval(sess, self.val_imgs[:30], self.val_labs[:30], 30, ops)
        pass;


    def test_compile_151(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 120)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        tf_simple_resnet.create_session()

    def test_training_151(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 10)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        sess = tf_simple_resnet.create_session()
        tf_simple_resnet.training(sess, 5, self.val_imgs, self.val_labs, 30, ops)

    def test_testing_151(self):
        tf.reset_default_graph()
        ops = tf_simple_resnet.resnet_34([None, 224, 224, 3], 10)
        tf_simple_resnet.compile('sgd', ops, 0.01)
        sess = tf_simple_resnet.create_session()
        tf_simple_resnet.eval(sess, self.val_imgs[:30], self.val_labs[:30], 30, ops)
        pass;
