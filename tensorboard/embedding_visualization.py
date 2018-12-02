#  임베딩은 고차원 벡터의 변환을 통해 생성할 수 있는 상대적인 저차원 공간을 가리킵니다
# ref : https://github.com/rmeertens/Simplest-Tensorflow-Tensorboard-MNIST-Embedding-Visualisation/blob/master/Minimal%20example%20embeddings.ipynb
import keras
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# DataSet
mnist = input_data.read_data_sets('MNIST_data' , one_hot=False)
n_embedding_data = 1000
batch_xs , batch_ys = mnist.train.next_batch(n_embedding_data)
logs_dir = './metadata/logs'

# Define Writer
summary_writer = tf.summary.FileWriter(logdir=logs_dir)

# Define embedding Variable
embedding_var = tf.Variable(tf.stack(mnist.test.images[:n_embedding_data] , axis=0), trainable=False)

# meta data path
metadata_path = './metadata/metadata.tsv'

# Embedding Projection Definition
from tensorflow.contrib.tensorboard.plugins import projector
#
work_dir = './metadata'

projector_config = projector.ProjectorConfig()
embedding_projection = projector_config.embeddings.add()
embedding_projection.tensor_name = embedding_var.name
embedding_projection.metadata_path = metadata_path
embedding_projection.sprite.image_path = os.path.join(work_dir + '/mnist_10k_sprite.png')
embedding_projection.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(summary_writer , projector_config)


# Create Session and Init global Variable
sess = tf.Session();
sess.run(tf.global_variables_initializer())
# Create Summary writer for tensorboard

projector.visualize_embeddings(summary_writer , projector_config)
tf.train.Saver().save(sess , './metadata/model.ckpt', global_step=1)


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1 - mnist_digits

to_visualise = batch_xs
to_visualise = vector_to_matrix_mnist(to_visualise)
to_visualise = invert_grayscale(to_visualise)

sprite_image = create_sprite_image(to_visualise)

plt.imsave(embedding_projection.sprite.image_path,sprite_image,cmap='gray')
plt.imshow(sprite_image,cmap='gray')


with open(metadata_path,'w') as f:
    f.write("Index\tLabel\n")
    for index,label in enumerate(batch_ys):
        f.write("%d\t%d\n" % (index,label))