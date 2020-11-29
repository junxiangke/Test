import tensorflow as tf

x = tf.Variable([1, 2, 3])
x[0] = 10
print(x)