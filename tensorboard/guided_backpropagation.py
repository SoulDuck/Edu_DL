import keras
image_width , image_height= 128, 128
vgg_model = keras.applications.vgg16.VGG16(include_top=False)

input_image = vgg_model.input
vgg_layer_dict = dict([(vgg_layer.name , vgg_layer) for vgg_layer in vgg_model.layers[:1]])
vgg_layer_output = vgg_layer_dict['block5_conv1'].output

