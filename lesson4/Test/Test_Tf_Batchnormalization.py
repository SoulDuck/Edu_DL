import sys
sys.path.append('../')
sys.path.append('../../')
import unittest
import cifar
import tensorflow as tf
import tf_simple_resnet
from PIL import Image
import numpy as np

class TestTFBatchNormalization(unittest.TestCase):
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


    def test_batchnormalization(self):
        """
        batch normalization 에 대해서 Test 합니다.
        :return:
        """
        tf.reset_default_graph()

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

        def apply_BN(x, y, phase_train):
            with tf.variable_scope('apply_bn'):
                he_init = tf.initializers.variance_scaling(scale=1.2)
                # layer 1
                layer = tf.layers.conv2d(x, 64, 3, 2, kernel_initializer=he_init)  # 32,32
                layer = tf_simple_resnet.batch_normalization(layer, phase_train ,'conv1')
                layer = tf.nn.relu(layer)
                # layer 2
                layer = tf.layers.conv2d(layer, 64, 3, 2, kernel_initializer=he_init)  # 32,32
                layer = tf_simple_resnet.batch_normalization(layer, phase_train, 'conv2')
                layer = tf.nn.relu(layer)
                # layer 3
                layer = tf.layers.conv2d(layer, 64, 3, 2, kernel_initializer=he_init)  # 32,32
                layer = tf_simple_resnet.batch_normalization(layer, phase_train, 'conv3')
                layer = tf.nn.relu(layer)
                # layer 4
                layer = tf.layers.conv2d(layer, 64, 3, 2, kernel_initializer=he_init)  # 32,32
                layer = tf_simple_resnet.batch_normalization(layer, phase_train, 'conv4')
                layer = tf.nn.relu(layer)

                h, w = layer.get_shape()[1:3]
                gap_layer = tf.layers.average_pooling2d(layer, (h, w), strides=(1, 1))
                flat_layer = tf.layers.flatten(gap_layer)
                # apply BN Graph
                xavier_init = tf.initializers.variance_scaling(scale=1)
                with tf.variable_scope('logits'):
                    logits = tf.layers.dense(flat_layer, 10, activation=None, kernel_initializer=xavier_init,
                                             use_bias=True)

                # Mean cost values
                costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
                cost_op = tf.reduce_mean(costs_op)

                # Accuracy
                y_cls = tf.argmax(y, axis=1)
                correct = tf.nn.in_top_k(logits, y_cls, 1)
                acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

                # Train_op
                train_op = tf.train.AdamOptimizer().minimize(cost_op)

                ops = {'cost_op': cost_op, 'acc_op': acc_op, 'train_op': train_op}

        def without_BN(x, y, phase_train):
            with tf.variable_scope('without_bn'):
                he_init = tf.initializers.variance_scaling(scale=1.2)

                layer = tf.layers.conv2d(x, 64, 3, 2, kernel_initializer=he_init, activation=tf.nn.relu)  # 32,32
                layer = tf.layers.conv2d(layer, 64, 3, 2, kernel_initializer=he_init, activation=tf.nn.relu)  # 32,32
                layer = tf.layers.conv2d(layer, 64, 3, 2, kernel_initializer=he_init, activation=tf.nn.relu)  # 32,32
                layer = tf.layers.conv2d(layer, 64, 3, 2, kernel_initializer=he_init, activation=tf.nn.relu)  # 32,32
                h, w = layer.get_shape()[1:3]
                gap_layer = tf.layers.average_pooling2d(layer, (h, w), strides=(1, 1))
                flat_layer = tf.layers.flatten(gap_layer)

                # apply BN Graph
                xavier_init = tf.initializers.variance_scaling(scale=1)
                with tf.variable_scope('logits'):
                    logits = tf.layers.dense(flat_layer, 10, activation=None, kernel_initializer=xavier_init,
                                             use_bias=True)

                # Mean cost values
                costs_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
                cost_op = tf.reduce_mean(costs_op)

                # Accuracy
                y_cls = tf.argmax(y, axis=1)
                correct = tf.nn.in_top_k(logits, y_cls, 1)
                acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

                # Train_op
                train_op = tf.train.AdamOptimizer().minimize(cost_op)

                ops = {'cost_op': cost_op, 'acc_op': acc_op, 'train_op': train_op}
                return ops

        """
        you can check applied batch normalization network result and
         without batch normalization network result
        """
        #
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], 'x')
        y = tf.placeholder(tf.float32, [None, 10], 'y')
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        BN_ops = apply_BN(x, y, phase_train)
        without_BN_ops = without_BN(x, y, phase_train)

        sess = tf.Session()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        apply_bn_costs = []
        apply_bn_accs = []
        without_bn_costs = []
        without_bn_accs = []

        # Fetches
        fetches = [BN_ops['train_op'], BN_ops['cost_op'], BN_ops['acc_op'],
                   without_BN_ops['train_op'], without_BN_ops['cost_op'], without_BN_ops['acc_op']]

        for i in range(10):
            batch_xs, batch_ys = cifar.next_batch(self.train_imgs, self.train_labs, 64)
            # Feed dicts
            feed_dict = {x: batch_xs, y: batch_ys, phase_train: True}

            _, bn_cost, bn_acc, _, without_bn_cost, without_bn_acc = sess.run(fetches, feed_dict)
            apply_bn_costs.append(bn_cost)
            apply_bn_accs.append(bn_acc)
            without_bn_costs.append(without_bn_cost)
            without_bn_accs.append(without_bn_acc)

        # Test Feed dicts
        feed_dict = {x: self.test_imgs, y: self.test_labs, phase_train: False}
        # Test fetches
        fetches = [BN_ops['cost_op'], BN_ops['acc_op'], without_BN_ops['cost_op'], without_BN_ops['acc_op']]
        bn_cost, bn_acc, without_bn_cost, without_bn_acc = sess.run(fetches, feed_dict)

        apply_bn_test_acc = bn_cost
        apply_bn_test_cost = bn_acc
        without_bn_test_acc = without_bn_cost
        without_bn_test_cost = without_bn_acc



