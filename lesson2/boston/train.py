import tensorflow as tf
import pandas as pd
import numpy as np
# boston
df = pd.read_csv('./train.csv')
train_datum = df[:-60].values
val_datum = df[-60:].values


def split_data_label(datum):
    xs = datum[:, 1:]
    ys = datum[:, -1:]
    return xs, ys

train_datum, train_labels = split_data_label(train_datum)
val_datum, val_labels = split_data_label(val_datum)

print('train data shape : {}'.format(train_datum.shape))
print('train labels shape : {}'.format(train_labels.shape))
print('validation data shape : {}'.format(val_datum.shape))
print('validation data shape : {}'.format(val_labels.shape))


def normalize(a, min_values ,max_values):
    return (a - min_values) / (max_values - min_values)


# noramlize train datum
min_values = np.min(train_datum, axis=0)
max_values = np.max(train_datum, axis=0)
norm_train_datum = normalize(train_datum , min_values, max_values)
norm_val_datum = normalize(val_datum , min_values, max_values)
print('Validation : ', np.min(norm_val_datum ), np.max(norm_val_datum))


# normalize train label
norm_train_labels = \
    (train_labels - min_values) / (max_values - min_values)

norm_val_labels = \
    (val_labels - min_values) / (max_values - min_values)

print(np.min(norm_train_labels ), np.max(norm_val_labels))

x = tf.placeholder(dtype=tf.float32, shape=[None, 14], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
lr = tf.placeholder(dtype=tf.float32, shape=[])

layer0 = tf.layers.dense(x, units=5, activation=tf.nn.sigmoid)
layer1 = tf.layers.dense(layer0, units=5, activation=tf.nn.sigmoid)
layer2 = tf.layers.dense(layer1, units=1, activation=tf.nn.sigmoid)

logits = tf.identity(layer2, name='logits')
loss = tf.losses.mean_squared_error(labels=y ,predictions=logits)

acc = tf.reduce_mean(tf.cast(tf.equal(y,(logits + tf.constant(0.5))) , tf.float32))

train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(50000):
    _, loss_ = sess.run([train_op, loss],
                        feed_dict={x: norm_train_datum, y: norm_train_labels , lr: 0.1})
    __, test_loss_ = sess.run([train_op, loss],
                            feed_dict={x: norm_val_datum, y: norm_val_labels , lr: 0.1})

    print(loss_, test_loss_)


# mission tensorboard
# model save and restore
# test dataset 확인 및 y 값 원상 복귀
# 정답하고 예측하고 비교하는 데이터 그려오기








