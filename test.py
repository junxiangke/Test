import tensorflow as tf
import numpy as np


class MyPointnetLayer(tf.keras.layers.Layer):
    def __init__(self, output_dims, **kwargs):
        self.output_dims = output_dims
        super(MyPointnetLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(1, kernel_size=(1, 3))
        super(MyPointnetLayer, self).build(input_shape)

    def call(self, x):
        y = self.conv_1(x)
        return y


def loss(x, y):
    return (x - y)**2


def train_network(data_x, data_y, model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    with tf.GradientTape() as tape:
        y_pred = model(data_x)
        grads = tape.gradient(loss(y_pred, data_y), model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


if __name__ == "__main__":
    # inputs = tf.keras.Input(shape=(3,))
    # my_net = MyPointnetLayer(1)
    # x = np.ones(shape=(1, 1, 3, 1), dtype=np.float32)
    # print(my_net.get_weights())
    # print(my_net(x))
    # print('train variable: ', my_net.trainable_variables)

    x = np.ones(shape=(1, 1, 3, 1), dtype=np.float32)
    model = MyPointnetLayer(1)
    model(x)
    train_network(x, x, model)