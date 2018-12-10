import sys
sys.path.append('../')
sys.path.append('../../')
import unittest
import cifar
import tensorflow as tf
import tf_alexnet
import os
from PIL import Image
import numpy as np
import extract
from sklearn.pipeline import Pipeline
from load import DogDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from transform import (Normalization, RandomFlip,
                       RandomRescaleAndCrop, RandomRotation,
                       RandomColorShift)


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

        # Dog Breed dataset
        # Setting extractor
        # Mock up (12 classes, 2848 dataset )
        self.dex = extract.DogExtractor('../data/dog_breed')

        # Setting Pipeline
        randomflip = RandomFlip()
        randomrescale = RandomRescaleAndCrop(max_ratio=1.3)
        randomrotation = RandomRotation(max_degree=30)
        randomshift = RandomColorShift(max_shift=20)
        normalization = Normalization()  # 0~255 -> 0~1로 정규화

        self.train_pipeline = Pipeline([
            ("랜덤으로 좌우 반전", randomflip),
            ("랜덤으로 rescale 후 crop", randomrescale),
            ("랜덤으로 rotation", randomrotation),
            ("랜덤으로 RGB color의 값을 shift", randomshift),
            ('0~1 범위로 정규화', normalization)
        ])

        # Splite Dataset Validation Datset
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
        self.train_index, self.test_index = next(split.split(self.dex.index, self.dex.labels))

        # Loader
        self.loader = DogDataGenerator(self.dex , self.train_index , self.train_pipeline, batch_size=64)

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

    def test_create_saver(self):
        """
        saver 파일이 잘 작동하나 확인합니다
         - 폴더생성 확인
         - 변수 저장 후 복원 확인
        :return:

        """
        init = tf.constant(1,shape=[2,3], name='var1')
        var = tf.Variable(init)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        src_var = sess.run(var)

        saver_dir= './tmp_dir/saver'
        saver_name = 'tmp_model'
        saver_path = os.path.join(saver_dir , saver_name)


        # 폴더 생성 확인
        saver = tf_alexnet.create_saver(saver_dir)
        self.assertIs(os.path.isdir(saver_dir) ,True)

        # 변수 저장
        saver.save(sess, save_path = saver_path)

        # 변수 저장 확인
        tf.reset_default_graph()
        new_saver = tf.train.import_meta_graph(os.path.join(saver_path+'.meta'))
        var = tf.get_default_graph().get_tensor_by_name('var1:0')
        new_saver.restore(sess, saver_path)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        dst_var = sess.run(var)

        self.assertListEqual(list(np.shape(src_var)), list(np.shape(dst_var)))
        self.assertEqual(np.sum(src_var), 6)
        self.assertEqual(np.sum(dst_var), 6)
        os.remove(saver_dir)

    def test_create_session(self):
        """
        Session 과 Variable Initalization 이 잘되는지 확인합니다.
        :return:
        """
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        tf_alexnet.compile('adagrad', ops, learning_rate=0.01)
        tf_alexnet.create_session()
        tf.reset_default_graph()

    def test_training(self):
        """
        트레이닝이 잘 되는지 확인합니다.

        :return:
        """
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=120)
        ops = tf_alexnet.compile('adagrad', ops, learning_rate=0.01)

        # Create session
        # Add train_op to ops
        sess = tf_alexnet.create_session()

        # Create Logger
        logger_dir= './tmp_dir/logger'
        logger = tf_alexnet.create_logger(logger_dir)

        # Training
        g_step = tf_alexnet.training(sess, self.loader, ops, logger , global_step = 1, n_iter = 10)
        g_step = tf_alexnet.training(sess, self.loader, ops, logger , global_step = g_step, n_iter = 10)

        # Reset tensorflow graph
        tf.reset_default_graph()
        os.remove(logger_dir)

    def test_eval(self):
        """
        Evaluation 하는 코드를 검증하니다

        :return:
        """
        ops = tf_alexnet.alexnet((None, 224, 224, 3), n_classes=10)
        ops = tf_alexnet.compile('adagrad', ops, learning_rate=0.01)

        # Create Session
        # Add train_op to ops
        sess = tf_alexnet.create_session()

        # Create Logger
        logger_dir= './tmp_dir/logger'
        logger = tf_alexnet.create_logger(logger_dir)

        # Create Saver
        saver_dir = './tmp_dir/saver'
        saver = tf_alexnet.create_saver(saver_dir)

        # training
        tf_alexnet.eval(sess, self.val_imgs[:60], self.val_labs[:60], ops=ops, logger=logger, saver=saver,
                        global_step=0)
        tf_alexnet.eval(sess, self.val_imgs[:60], self.val_labs[:60], ops=ops, logger=logger, saver=saver,
                        global_step=100)
        tf_alexnet.eval(sess, self.val_imgs[:60], self.val_labs[:60], ops=ops, logger=logger, saver=saver,
                        global_step=200)
        #
        tf.reset_default_graph()

    def tearDown(self):
        pass;