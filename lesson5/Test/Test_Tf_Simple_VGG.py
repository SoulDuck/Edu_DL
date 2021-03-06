import sys
sys.path.append('../')
sys.path.append('../../')
import unittest
from DataExtractor import cifar
import tensorflow as tf
import tf_simple_vgg


class TestTFSimpleVGG(unittest.TestCase):
    def setUp(self):
        # Cifar10 dataprovider
        cifar10 = cifar.Cifar10()
        self.train_imgs = cifar10.train_imgs
        self.train_labs = cifar10.train_labs
        self.test_imgs = cifar10.test_imgs
        self.test_labs = cifar10.test_labs
        self.val_imgs = cifar10.val_imgs
        self.val_labs = cifar10.val_labs

    def test_vgg11(self):
        """
        알렉스넷이 잘 구성되었는지 학인합니다.
        :return:
        """
        tf.reset_default_graph()
        tf_simple_vgg.vgg_11((None, 32, 32, 3), n_classes=10)
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        learning_rate = tf.get_default_graph().get_tensor_by_name('learning_rate:0')
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')

        # Train Filewriter
        model_name = 'vgg_11'
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
        feed_dict = {x: self.train_imgs[:60], y: self.train_labs[:60], phase_train: True, learning_rate: 0.1,
                     keep_prob: 0.5}

        # Training
        for step in range(10):
            _, cost, acc, summary_merge = sess.run(fetches=fetches, feed_dict=feed_dict)
            writer.add_summary(summary_merge, global_step=step)
            print('step : {} cost : {} , acc : {}'.format(step, cost, acc))

        # Validation
        fetches = [cost_op, acc_op]
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False, keep_prob: 0.5}
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
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        fetches = [acc_op, cost_op]

        # Run Session
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False, keep_prob: 0.5}
        restore_acc, restore_cost = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Restore 했을때 Validation 결과값이 동일해야 합니다
        self.assertEqual(float(restore_acc), float(val_acc))
        self.assertEqual(float(restore_cost), float(val_cost))

    def test_vgg13(self):
        """
        알렉스넷이 잘 구성되었는지 학인합니다.
        :return:
        """
        tf.reset_default_graph()
        tf_simple_vgg.vgg_13((None, 32, 32, 3), n_classes=10)
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        learning_rate = tf.get_default_graph().get_tensor_by_name('learning_rate:0')
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')

        # Train Filewriter
        model_name = 'vgg_13'
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
        feed_dict = {x: self.train_imgs[:60], y: self.train_labs[:60], phase_train: True, learning_rate: 0.01,
                     keep_prob: 0.5}

        # Training
        for step in range(5):
            _, cost, acc, summary_merge = sess.run(fetches=fetches, feed_dict=feed_dict)
            writer.add_summary(summary_merge, global_step=step)
            print('step : {} cost : {} , acc : {}'.format(step, cost, acc))

        # Validation
        fetches = [cost_op, acc_op]
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False, keep_prob: 0.5}
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
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        fetches = [acc_op, cost_op]

        # Run Session
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False, keep_prob: 0.5}
        restore_acc, restore_cost = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Restore 했을때 Validation 결과값이 동일해야 합니다
        self.assertEqual(float(restore_acc), float(val_acc))
        self.assertEqual(float(restore_cost), float(val_cost))

    def test_vgg16(self):
        """
        알렉스넷이 잘 구성되었는지 학인합니다.
        :return:
        """
        tf.reset_default_graph()
        tf_simple_vgg.vgg_16((None, 32, 32, 3), n_classes=10)
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        learning_rate = tf.get_default_graph().get_tensor_by_name('learning_rate:0')
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')

        # Train Filewriter
        model_name = 'vgg_16'
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
        feed_dict = {x: self.train_imgs[:60], y: self.train_labs[:60], phase_train: True, learning_rate: 0.0001,
                     keep_prob: 0.5}

        # Training
        for step in range(5):
            _, cost, acc, summary_merge = sess.run(fetches=fetches, feed_dict=feed_dict)
            writer.add_summary(summary_merge, global_step=step)
            print('step : {} cost : {} , acc : {}'.format(step, cost, acc))

        # Validation
        fetches = [cost_op, acc_op]
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False, keep_prob: 0.5}
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
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        fetches = [acc_op, cost_op]

        # Run Session
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False, keep_prob: 0.5}
        restore_acc, restore_cost = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Restore 했을때 Validation 결과값이 동일해야 합니다
        self.assertEqual(float(restore_acc), float(val_acc))
        self.assertEqual(float(restore_cost), float(val_cost))

    def test_vgg19(self):
        """
        알렉스넷이 잘 구성되었는지 학인합니다.
        :return:
        """
        tf.reset_default_graph()
        tf_simple_vgg.vgg_19((None, 32, 32, 3), n_classes=10)
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        phase_train = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        learning_rate = tf.get_default_graph().get_tensor_by_name('learning_rate:0')
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')

        # Train Filewriter
        model_name = 'vgg_19'
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
        feed_dict = {x: self.train_imgs[:60], y: self.train_labs[:60], phase_train: True, learning_rate: 0.0001,
                     keep_prob: 0.5}

        # Training
        for step in range(5):
            _, cost, acc, summary_merge = sess.run(fetches=fetches, feed_dict=feed_dict)
            writer.add_summary(summary_merge, global_step=step)
            print('step : {} cost : {} , acc : {}'.format(step, cost, acc))

        # Validation
        fetches = [cost_op, acc_op]
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False, keep_prob: 0.5}
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
        keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        fetches = [acc_op, cost_op]

        # Run Session
        feed_dict = {x: self.val_imgs[:60], y: self.val_labs[:60], phase_train: False, keep_prob: 0.5}
        restore_acc, restore_cost = sess.run(fetches=fetches, feed_dict=feed_dict)

        # Restore 했을때 Validation 결과값이 동일해야 합니다
        self.assertEqual(float(restore_acc), float(val_acc))
        self.assertEqual(float(restore_cost), float(val_cost))
