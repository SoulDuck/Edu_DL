import unittest
import sys
sys.path.append('../')
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages')
from extract import DogExtractor
from load import DogDataGenerator
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras_alexnet import alexnet , logits , training
import configure as cfg
import cifar


class TestKerasAlexnet(unittest.TestCase):
    def setUp(self):
        dex = DogExtractor('../../data/dog_breed')
        self.doggen = DogDataGenerator(dex)

    def test_keras_alexnet(self):
        """
        [ O ] Alexnet 의 구조를 확인합니다

        """
        x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
        top_conv = alexnet(x)
        pred = logits(top_conv)
        training(x, pred , self.doggen)

    def test_check_tensorboard_graph(self):
        """
        - Tensorboard 그래프를 그리고 구조가 잘 작성 되어 있는지 확인합니다
        - Validation Loss Graph , Validation Accuracy Graph
        - Train Loss Graph , Train Accuracy Graph

        :return:
        """
    def test_keras_alexnet_cifar(self):
        """

        :return:
        """
        cifar10 = cifar.Cifar10()
        train_imgs = cifar10.train_imgs
        train_labs = cifar10.train_labs
        x = Input(shape=(cfg.img_h, cfg.img_w, cfg.img_ch))
        top_conv = alexnet(x)
        pred = logits(top_conv)
        model = Model(x, pred)
        model.summary()
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mse', 'accuracy'])
        model.fit(train_imgs , train_labs ,32)

    def tearDown(self):
        pass;