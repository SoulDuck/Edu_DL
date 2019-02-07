import tensorflow as tf

'a' = 0
'b' = 1
'c' = 2


#

0

a = tf.one_hot(2 ,depth =3)
sess = tf.Session()
print(sess.run(a))


