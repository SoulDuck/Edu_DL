class MnistLoader(object):
    def __init__(self):
        self.img_size = 28
        self.num_channels = 1
        # download MNIST data
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        self.train_imgs = mnist.train.images
        self.train_labs = mnist.train.labels
        self.test_imgs = mnist.test.images
        self.test_labs = mnist.test.labels
        self.val_imgs = mnist.validation.images
        self.val_labs = mnist.validation.labels

        self.train_imgs, self.test_imgs, self.val_imgs = map(lambda imgs: imgs.reshape([-1, 28, 28, 1]),
                                                             [self.train_imgs, self.test_imgs, self.val_imgs])



MnistLoader().train_imgs