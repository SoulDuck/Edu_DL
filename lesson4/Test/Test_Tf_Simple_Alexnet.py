import sys
sys.path.append('../')
sys.path.append('../../')
import unittest
import cifar
import tensorflow as tf
import tf_simple_alexnet
from PIL import Image
import numpy as np


class TestTFSimpleAlexnet(unittest.TestCase):
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

    def test_conv(self):
        """
        Convolution filter 가 잘 구성되어 있는지 확인합니다
        :return: 
        """

        # kernel
        kernel = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
        kernel[:, 3, 0, 0] = 1
        kernel[3, :, 0, 1] = 1

        # Input Image
        sample_imgs = self.val_imgs[0:1]
        self.assertIs(np.ndim(sample_imgs), 4)
        sample_imgs_node = tf.Variable(sample_imgs, dtype=tf.float32)

        # Convolution Function
        he_init = tf.initializers.variance_scaling(scale=2)
        layer = tf.layers.conv2d(sample_imgs_node, filters=96, kernel_size=11, strides=4, padding='same',
                                 kernel_initializer=he_init,
                                 activation=tf.nn.relu, use_bias=True)

        # Checking
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.close()
        tf.reset_default_graph()

        self.assertListEqual(list(np.shape(layer)), [1, 56, 56, 96])

    def test_fc(self):
        """
        tf.layers.dense 가 잘 작동하는지 확인합니다.
        :return:
        """

        # kernel
        kernel = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
        kernel[:, 3, 0, 0] = 1
        kernel[3, :, 0, 1] = 1

        # Input Image
        sample_imgs = self.val_imgs[0:1]
        self.assertIs(np.ndim(sample_imgs), 4)
        sample_imgs_node = tf.Variable(sample_imgs, dtype=tf.float32)

        # Flatten layer
        flat_layer = tf.layers.flatten(sample_imgs_node)

        # Fully connected layer 가 잘 작동하는 지 확인합니다.
        xavier_init = tf.initializers.variance_scaling(scale=1)
        with tf.variable_scope('fc1'):
            layer = tf.layers.dense(flat_layer, 1000, activation=tf.nn.relu, kernel_initializer=xavier_init,
                                    use_bias=True)

        # Session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.close()
        tf.reset_default_graph()

        # 이미지가 1장이고 output node 가 1000 임으로 [1,1000] 으로 나와야 합니다
        self.assertListEqual(list(np.shape(layer)), [1, 1000])

    def test_top_n_k(self):
        """
        accuracy을 구하는 tf.nn.top_n_k 에 대해 알아봅니다
        :return: 
        """
        # Mock Up data
        logits = [[0.05, 0.9, 0.05],
                  [0.05, 0.9, 0.05],
                  [0.05, 0.9, 0.05],
                  [0.05, 0.9, 0.05]]
        y = [1, 1, 2, 1]
        logits_node = tf.Variable(logits)
        y_node = tf.Variable(y)
        
        correct = tf.nn.in_top_k(logits_node, y_node, 1)
        acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        acc = sess.run(acc_op)

        self.assertIs(acc, 0.75)

    def test_alexnet(self):
        """
        알렉스넷이 잘 구성되었는지 학인합니다.
        :return:
        """
        tf_simple_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf.reset_default_graph()

    def test_compile_sgd(self):
        """
        GradientDescent 잘 작동하는지 확인합니다
        :return:
        """
        ops = tf_simple_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_simple_alexnet.compile('sgd', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_compile_momentum(self):
        ops = tf_simple_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_simple_alexnet.compile('momentum', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_compile_rmsprop(self):
        ops = tf_simple_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_simple_alexnet.compile('rmsprop', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_compile_adadelta(self):
        ops = tf_simple_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_simple_alexnet.compile('adadelta', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_compile_adagrad(self):
        ops = tf_simple_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_simple_alexnet.compile('adagrad', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_compile_adam(self):
        ops = tf_simple_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_simple_alexnet.compile('adam', ops, learning_rate=0.01)
        tf.reset_default_graph()

    def test_create_session(self):
        """
        Session 과 Variable Initalization 이 잘되는지 확인합니다.
        :return:
        """
        tf.reset_default_graph()
        ops = tf_simple_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_simple_alexnet.compile('adagrad', ops, learning_rate=0.01)
        tf_simple_alexnet.create_session('alexnet')

    def test_training(self):
        """
        Training 이 잘 작동하는지 확인합니다
        :return:
        """
        tf.reset_default_graph()
        ops = tf_simple_alexnet.alexnet((None, 224, 224, 3), n_classes=10)
        ops = tf_simple_alexnet.compile('adagrad', ops, learning_rate=0.01)

        # Create session
        # Add train_op to ops
        sess, saver, writer = tf_simple_alexnet.create_session('alexnet')

        # Training
        cost = tf_simple_alexnet.training(sess, 1, self.val_imgs[:60], self.val_labs[:60], ops=ops)
        self.assertIsInstance(cost, list)

        # Reset tensorflow graph

    def test_eval(self):
        """
        :return:
        """
        tf.reset_default_graph()
        ops = tf_simple_alexnet.alexnet((None, 224, 224, 3), n_classes=10)
        ops = tf_simple_alexnet.compile('adagrad', ops, learning_rate=0.01)

        # add train_op to ops
        # create session
        sess, saver, writer = tf_simple_alexnet.create_session('alexnet')

        # training
        acc, cost = tf_simple_alexnet.eval(sess, self.val_imgs[:60], self.val_labs[:60], ops=ops)

        self.assertIsInstance(acc, np.float32)
        self.assertIsInstance(cost, np.float32)

    def tearDown(self):
        pass
