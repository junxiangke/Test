import tensorflow as tf
import numpy as np

x = tf.compat.v1.random_normal(shape=(3, 4),mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
print(x)
res = tf.reduce_sum(x)
print(res)
