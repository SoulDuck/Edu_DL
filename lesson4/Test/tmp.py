import tensorflow as tf

#init
phase_train = tf.placeholder(tf.bool)
init=tf.constant(-1,shape=[3])
# get variable
a=tf.get_variable('bias' ,initializer=init)

#output = tf.cond(phase_train , lambda: tf.nn.relu(a) ,lambda: a)

def fn(input, activation):
    return activation(input)


output = fn(a, None)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(output , feed_dict={phase_train : True}))