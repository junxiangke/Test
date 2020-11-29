import tensorflow as tf
import numpy as np

def train_network(data_x):
    # with tf.GradientTape() as tape:
    #     model()
    pass

def model(x):
    return tf.Variable(10, trainable=True) * x

if __name__ == "__main__":
    data_x = np.arange(10)
    data_y = model(data_x)
    print(data_y)
    print(model)


