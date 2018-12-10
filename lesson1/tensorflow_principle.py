# 학습 목표
# - 노드 개념 이해하기
# - 변수의 개념과 변수를 만드는 법
# - 변수를 관리 하기
# - Tensorflow 의 Namespace 이해하기

# - 궁굼한 것들 각  Variable 이 어떤 Graph Key 에 들어 있는지 확인할 려면?
# - Input placeholder 는 기본적으로 어떤 그래프에 들어가지?
# - c

import tensorflow as tf
with tf.variable_scope('placholder') as name:
    phase_train = tf.placeholder(tf.bool, name = 'phase_train')

init_a = tf.constant(0.0,shape=[1,32,32,3])
init_b = tf.constant(0.0 , shape=[1])
tf.add_to_collection('my_collections' ,value=phase_train)

var1 = tf.Variable(init_a, trainable=False)
var2 = tf.Variable(init_b, trainable=True)
y = tf.placeholder(dtype=tf.float32 , shape=[1])
cost = y- var2




train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
print(train_op)
#var1 = tf.Variable(init_a, collections=['my_collections',tf.GraphKeys.GLOBAL_VARIABLES])
#conv1 = tf.layers.conv2d(var1,3,3,2)
#global_step = tf.Variable(0,trainable=False)
acc_op = tf.metrics.accuracy(labels=var2, predictions=y)
print(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
print(tf.get_collection(tf.GraphKeys.TRAIN_OP))
print(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES))
print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
print(tf.get_collection('my_collections'))
vars=tf.get_collection('my_collections')
print(vars[0].op.name)


sess=tf.Session()
sess.run(tf.global_variables_initializer())


