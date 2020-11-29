import tensorflow as tf
import numpy as np

conv = tf.keras.layers.Conv2D(1, kernel_size=(1, 3))
x = np.ones(shape=(1, 1, 3, 1), dtype=np.float32)
y = conv(x)
print(y)
weight = conv.get_weights()
print(np.sum(weight[0]))
