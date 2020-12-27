import tensorflow as tf
import numpy as np

class MyConv(tf.keras.layers.Layer):
    def __init__(self):
        self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1))
        self.conv_2 = tf.keras.layers.Conv2D(filters=256)
        self.conv_out = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1))

    def call(self, inp_pos, out_pos, inp_features, neighbors_index):
        embedding = tf.nn.embedding_lookup(inp_features, neighbors_index)
        inp_pos_1 = tf.concat([tf.zeros((1, 3)), inp_pos], axis=0)
        pos_feat = tf.nn.embedding_lookup(inp_pos_1, neighbors_index)
        x = tf.concat([inp_pos_1, pos_feat], axis=-1)
        x = tf.expand_dims(x, axis=1)
        x = self.conv_1(x)
        x = tf.keras.activations.relu(x)
        x = self.conv_2(x)
        x = tf.reduce_sum(x, axis=2, keepdims=True)
        out_feat = self.conv_out(x)
        return out_feat



        x = tf.range(24)
        x = tf.reshape(x, shape=(1, 2, 3, 4))
        print(x)
        y = tf.transpose(x, perm=[0, 1, 3, 2])
        print(y)
