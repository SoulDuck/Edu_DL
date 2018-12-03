import __init__
from keras.models import Model
from keras.layers import Input , Dense , Conv2D , MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD
import glob
import configure as cfg

# Load cifar-10 dataset
"""
import cifar
train_filenames = glob.glob('./cifar_10/cifar-10-batches-py/data_batch*')
train_imgs, train_labs = cifar_10.get_images_labels(*train_filenames)
"""

# Load dogs dataset
import dog_breed
test_dir = '/Users/seongjungkim/PycharmProjects/alexnet/dog_breed'
dog_extractor = dog_breed.Dog_Extractor(test_dir ,2 )
train_imgs = dog_extractor.imgs
train_labs = dog_extractor.labs

# Load cifar-10 dataset
train_imgs = train_imgs/255.

# VGG 16
# Block 1

def VGG_block(kernel_sizes , out_chs, strides, input):
    """

    :param kernel_sizes: tuple , E.g) [3,3,3,3] , kernel size 가 (3,3)인 CNN을 4번 생성합니다
    :param input: 입력으로 넣을수 있는 Tensor
    :return:
    """
    layer = input
    ksizes_outchs_strides = zip(kernel_sizes, out_chs, strides)
    for k , ch , s in ksizes_outchs_strides:
        layer = Conv2D(ch, (k, k), strides=(s, s), padding='same', activation='relu')(layer)
    layer = MaxPooling2D()(layer)
    return layer


x=Input(shape=(cfg.img_h, cfg.img_w , cfg.img_ch))
"""
# VGG 11 Convnet
layer = VGG_block(kernel_sizes=[3] , out_chs=[64], strides=[1] , input=x)
layer = VGG_block(kernel_sizes=[3] , out_chs=[128], strides=[1] , input=layer)
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[256,256], strides=[1,1], input=layer)
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[512,512], strides=[1,1], input=layer)
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[512,512], strides=[1,1], input=layer)
"""

"""
# VGG 13 Convnet
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[64,64], strides=[1,1] , input=x)
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[128,128], strides=[1,1] , input=layer)
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[256,256], strides=[1,1], input=layer)
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[512,512], strides=[1,1], input=layer)
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[512,512], strides=[1,1], input=layer)
"""

"""
# VGG 16 Convnet
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[64,64], strides=[1,1] , input=x)
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[128,128], strides=[1,1] , input=layer )
layer = VGG_block(kernel_sizes=[3,3,3] , out_chs=[256,256,256], strides=[1,1,1], input=layer)
layer = VGG_block(kernel_sizes=[3,3,3] , out_chs=[512,512,512], strides=[1,1,1], input=layer)
layer = VGG_block(kernel_sizes=[3,3,3] , out_chs=[512,512,512], strides=[1,1,1], input=layer)
"""

"""
# VGG 19 Convnet
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[64,64], strides=[1,1] , input=x)
layer = VGG_block(kernel_sizes=[3,3] , out_chs=[128,128], strides=[1,1] , input=layer )
layer = VGG_block(kernel_sizes=[3,3,3,3] , out_chs=[256,256,256,256], strides=[1,1,1,1], input=layer)
layer = VGG_block(kernel_sizes=[3,3,3,3] , out_chs=[512,512,512,512], strides=[1,1,1,1], input=layer)
layer = VGG_block(kernel_sizes=[3,3,3,3] , out_chs=[512,512,512,512], strides=[1,1,1,1], input=layer)
"""

top_conv = layer
# FC Layer
flat_top_conv = Flatten()(top_conv)
fc1 = Dense(4096, activation='relu')(flat_top_conv)
fc1 = Dropout(0.5)(fc1)
fc2 = Dense(4096, activation='relu')(fc1)
fc2 = Dropout(0.5)(fc2)
pred = Dense(cfg.n_classes , activation='softmax')(fc2)
model = Model(x,pred)
model.summary()
# Training
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mse', 'accuracy'])
batch_size=30
epochs=10

model.fit(train_imgs, train_labs, batch_size=batch_size, epochs=epochs, verbose=1)

