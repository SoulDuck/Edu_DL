import tensorflow as tf

x=tf.placeholder(dtype=tf.float32 , shape = [ None , 32,32, 3 ] , name='input')

input_ch = 3
output_ch = 6
conv_filters  = tf.Variable(tf.random_normal(shape=[3,3,input_ch , output_ch] , stddev= 1.0))
conv_bias  = tf.Variable(tf.random_normal(shape=[output_ch] , stddev= 1.0))
conv_output = tf.nn.conv2d(x , conv_filters , strides = [1,1,1,1] , padding='SAME') + conv_bias

print(conv_output.get_shape())

print(conv_output)
sess=tf.Session()
init = tf.group(tf.global_variables_initializer())
sess.run(init)