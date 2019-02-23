import sys
sys.path.append('../')
sys.path.append('../../')
import unittest
import cifar
import tensorflow as tf
import tf_simple_resnet
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
        layer = tf_simple_resnet.bottlenect_block_projection(layer, 64, self.phase_train)

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

    def test_batch_norm(self):
        pass

    def test_resnet_18(self):
        """
        resnet 18 graph가 잘 그려지는지 확인합니다
        :return:
        """
        tf.reset_default_graph()

        tf_simple_resnet.resnet_18([None, 32, 32, 3], 10)
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        learning_rate = tf.get_default_graph().get_tensor_by_name('learning_rate:0')

        # Train Filewriter
        model_name = 'resnet_18'
        writer = tf.summary.FileWriter('tmp_dir/logs/{}'.format(model_name))
        writer.add_graph(tf.get_default_graph())

        # Define Operation
        cost_op = tf.get_default_graph().get_tensor_by_name('cost_op:0')
        acc_op = tf.get_default_graph().get_tensor_by_name('acc_op:0')
        # acc_op = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)[0]
        train_op = tf.get_collection(tf.GraphKeys.TRAIN_OP)[0]
        merge_op = tf.summary.merge_all()

        # Create Session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        fetches = [train_op, cost_op, acc_op, merge_op]
        feed_dict = {x: self.train_imgs[:60], y: self.train_labs[:60], phase_train: True, learning_rate: 0.01}

        # Training
        for step in range(5):
            _, cost, acc, summary_merge = sess.run(fetches=fetches, feed_dict=feed_dict)
            writer.add_summary(summary_merge, global_step=step)
            print('step : {} cost : {} , acc : {}'.format(step, cost, acc))

        # Validation
        fetches = [cost_op, acc_op]
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False}
        val_cost, val_acc = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Model save and restore
        saver = tf.train.Saver()
        saver.save(sess, './tmp_dir/models/{}/model'.format(model_name))

        # Reset graph
        tf.reset_default_graph()
        tf.train.import_meta_graph('./tmp_dir/models/{}/model.meta'.format(model_name))
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, './tmp_dir/models/{}/model'.format(model_name))

        # Restore Node
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        cost_op = tf.get_default_graph().get_tensor_by_name('cost_op:0')
        acc_op = tf.get_default_graph().get_tensor_by_name('acc_op:0')
        fetches = [acc_op, cost_op]

        # Run Session
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False}
        restore_acc, restore_cost = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Restore 했을때 Validation 결과값이 동일해야 합니다
        self.assertEqual(float(restore_acc), float(val_acc))
        self.assertEqual(float(restore_cost), float(val_cost))

    def test_resnet_34(self):
        """
        resnet 34 graph가 잘 그려지는지 확인합니다
        :return:
        """
        tf.reset_default_graph()

        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        learning_rate = tf.get_default_graph().get_tensor_by_name('learning_rate:0')

        # Train Filewriter
        model_name = 'resnet_34'
        writer = tf.summary.FileWriter('tmp_dir/logs/{}'.format(model_name))
        writer.add_graph(tf.get_default_graph())

        # Define Operation
        cost_op = tf.get_default_graph().get_tensor_by_name('cost_op:0')
        acc_op = tf.get_default_graph().get_tensor_by_name('acc_op:0')
        # acc_op = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)[0]
        train_op = tf.get_collection(tf.GraphKeys.TRAIN_OP)[0]
        merge_op = tf.summary.merge_all()

        # Create Session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        fetches = [train_op, cost_op, acc_op, merge_op]
        feed_dict = {x: self.train_imgs[:60], y: self.train_labs[:60], phase_train: True, learning_rate: 0.0001}

        # Training
        for step in range(5):
            _, cost, acc, summary_merge = sess.run(fetches=fetches, feed_dict=feed_dict)
            writer.add_summary(summary_merge, global_step=step)
            print('step : {} cost : {} , acc : {}'.format(step, cost, acc))

        # Validation
        fetches = [cost_op, acc_op]
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False}
        val_cost, val_acc = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Model save and restore
        saver = tf.train.Saver()
        saver.save(sess, './tmp_dir/models/{}/model'.format(model_name))

        # Reset graph
        tf.reset_default_graph()
        tf.train.import_meta_graph('./tmp_dir/models/{}/model.meta'.format(model_name))
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, './tmp_dir/models/{}/model'.format(model_name))

        # Restore Node
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        cost_op = tf.get_default_graph().get_tensor_by_name('cost_op:0')
        acc_op = tf.get_default_graph().get_tensor_by_name('acc_op:0')
        fetches = [acc_op, cost_op]

        # Run Session
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False}
        restore_acc, restore_cost = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Restore 했을때 Validation 결과값이 동일해야 합니다
        self.assertEqual(float(restore_acc), float(val_acc))
        self.assertEqual(float(restore_cost), float(val_cost))

    def test_resnet_50(self):
        """
        resnet 50 graph가 잘 그려지는지 확인합니다
        :return:
        """
        tf.reset_default_graph()
        tf_simple_resnet.resnet_50([None, 32, 32, 3], 10)
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        learning_rate = tf.get_default_graph().get_tensor_by_name('learning_rate:0')

        # Train Filewriter
        model_name = 'resnet_50'
        writer = tf.summary.FileWriter('tmp_dir/logs/{}'.format(model_name))
        writer.add_graph(tf.get_default_graph())

        # Define Operation
        cost_op = tf.get_default_graph().get_tensor_by_name('cost_op:0')
        acc_op = tf.get_default_graph().get_tensor_by_name('acc_op:0')
        train_op = tf.get_collection(tf.GraphKeys.TRAIN_OP)[0]
        merge_op = tf.summary.merge_all()

        # Create Session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        fetches = [train_op, cost_op, acc_op, merge_op]
        feed_dict = {x: self.train_imgs[:60], y: self.train_labs[:60], phase_train: True, learning_rate: 0.0001}

        # Training
        for step in range(5):
            _, cost, acc, summary_merge = sess.run(fetches=fetches, feed_dict=feed_dict)
            writer.add_summary(summary_merge, global_step=step)
            print('step : {} cost : {} , acc : {}'.format(step, cost, acc))

        # Validation
        fetches = [cost_op, acc_op]
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False}
        val_cost, val_acc = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Model save and restore
        saver = tf.train.Saver()
        saver.save(sess, './tmp_dir/models/{}/model'.format(model_name))

        # Reset graph
        tf.reset_default_graph()
        tf.train.import_meta_graph('./tmp_dir/models/{}/model.meta'.format(model_name))
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, './tmp_dir/models/{}/model'.format(model_name))

        # Restore Node
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        cost_op = tf.get_default_graph().get_tensor_by_name('cost_op:0')
        acc_op = tf.get_default_graph().get_tensor_by_name('acc_op:0')
        fetches = [acc_op, cost_op]

        # Run Session
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False}
        restore_acc, restore_cost = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Restore 했을때 Validation 결과값이 동일해야 합니다
        self.assertEqual(float(restore_acc), float(val_acc))
        self.assertEqual(float(restore_cost), float(val_cost))

    def test_resnet_101(self):
        """
        resnet 101 graph가 잘 그려지는지 확인합니다
        :return:
        """
        tf.reset_default_graph()
        tf_simple_resnet.resnet_101([None, 32, 32, 3], 10)
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        learning_rate = tf.get_default_graph().get_tensor_by_name('learning_rate:0')

        # Train Filewriter
        model_name = 'resnet_101'
        writer = tf.summary.FileWriter('tmp_dir/logs/{}'.format(model_name))
        writer.add_graph(tf.get_default_graph())

        # Define Operation
        cost_op = tf.get_default_graph().get_tensor_by_name('cost_op:0')
        acc_op = tf.get_default_graph().get_tensor_by_name('acc_op:0')
        # acc_op = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)[0]
        train_op = tf.get_collection(tf.GraphKeys.TRAIN_OP)[0]
        merge_op = tf.summary.merge_all()

        # Create Session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        fetches = [train_op, cost_op, acc_op, merge_op]
        feed_dict = {x: self.train_imgs[:60], y: self.train_labs[:60], phase_train: True, learning_rate: 0.000001}

        # Training
        for step in range(5):
            _, cost, acc, summary_merge = sess.run(fetches=fetches, feed_dict=feed_dict)
            writer.add_summary(summary_merge, global_step=step)
            print('step : {} cost : {} , acc : {}'.format(step, cost, acc))

        # Validation
        fetches = [cost_op, acc_op]
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False}
        val_cost, val_acc = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Model save and restore
        saver = tf.train.Saver()
        saver.save(sess, './tmp_dir/models/{}/model'.format(model_name))

        # Reset graph
        tf.reset_default_graph()
        tf.train.import_meta_graph('./tmp_dir/models/{}/model.meta'.format(model_name))
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, './tmp_dir/models/{}/model'.format(model_name))

        # Restore Node
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        cost_op = tf.get_default_graph().get_tensor_by_name('cost_op:0')
        acc_op = tf.get_default_graph().get_tensor_by_name('acc_op:0')
        fetches = [acc_op, cost_op]

        # Run Session
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False}
        restore_acc, restore_cost = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Restore 했을때 Validation 결과값이 동일해야 합니다
        self.assertEqual(float(restore_acc), float(val_acc))
        self.assertEqual(float(restore_cost), float(val_cost))

    def test_resnet_151(self):
        """
        resnet 151 graph가 잘 그려지는지 확인합니다
        :return:
        """
        tf.reset_default_graph()
        tf_simple_resnet.resnet_151([None, 32, 32, 3], 10)

        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        learning_rate = tf.get_default_graph().get_tensor_by_name('learning_rate:0')

        # Train Filewriter
        model_name = 'resnet_151'
        writer = tf.summary.FileWriter('tmp_dir/logs/{}'.format(model_name))
        writer.add_graph(tf.get_default_graph())

        # Define Operation
        cost_op = tf.get_default_graph().get_tensor_by_name('cost_op:0')
        acc_op = tf.get_default_graph().get_tensor_by_name('acc_op:0')
        # acc_op = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)[0]
        train_op = tf.get_collection(tf.GraphKeys.TRAIN_OP)[0]
        merge_op = tf.summary.merge_all()

        # Create Session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        fetches = [train_op, cost_op, acc_op, merge_op]
        feed_dict = {x: self.train_imgs[:60], y: self.train_labs[:60], phase_train: True, learning_rate: 0.000000001}

        # Training
        for step in range(5):
            _, cost, acc, summary_merge = sess.run(fetches=fetches, feed_dict=feed_dict)
            writer.add_summary(summary_merge, global_step=step)
            print('step : {} cost : {} , acc : {}'.format(step, cost, acc))

        # Validation
        fetches = [cost_op, acc_op]
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False}
        val_cost, val_acc = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Model save and restore
        saver = tf.train.Saver()
        saver.save(sess, './tmp_dir/models/{}/model'.format(model_name))

        # Reset graph
        tf.reset_default_graph()
        tf.train.import_meta_graph('./tmp_dir/models/{}/model.meta'.format(model_name))
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, './tmp_dir/models/{}/model'.format(model_name))

        # Restore Node
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        cost_op = tf.get_default_graph().get_tensor_by_name('cost_op:0')
        acc_op = tf.get_default_graph().get_tensor_by_name('acc_op:0')
        fetches = [acc_op, cost_op]

        # Run Session
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False}
        restore_acc, restore_cost = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Restore 했을때 Validation 결과값이 동일해야 합니다
        self.assertEqual(float(restore_acc), float(val_acc))
        self.assertEqual(float(restore_cost), float(val_cost))