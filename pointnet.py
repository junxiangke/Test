import tensorflow as tf
import numpy as np


class PointNet(tf.keras.Model):
    def __init__(self, filters):
        super(PointNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 3))

    def call(self, embedding, lookup_table):
        embedding = tf.concat([tf.zeros(shape=(1, embedding.shape[1])), embedding], axis=0)
        out = tf.nn.embedding_lookup(embedding, lookup_table)
        out = tf.expand_dims(out, axis=-1)
        out = self.conv(out)
        # print(out.shape)
        out = tf.reduce_max(out, axis=1)
        out = tf.reshape(out, shape=(lookup_table.shape[0], -1))
        return out


def create_data(embedding_shape=(4, 3), lookup_table_shape=(3, 5)):
    embedding = np.random.randn(*embedding_shape)
    embedding = embedding.astype(np.float32)
    lookup_table = np.random.randint(0, embedding_shape[0], lookup_table_shape)
    y_true = np.random.randn(lookup_table_shape[0], 3)
    y_true = y_true.astype(np.float32)
    return embedding, lookup_table, y_true


def create_model():
    model = PointNet(3)
    return model


def loss_fn(pred, y_true):
    loss = tf.reduce_sum((pred - y_true) ** 2)
    return loss


@tf.function(experimental_relax_shapes=True)
def train_model(model, inputs, y_true, optimizer):
    embedding, lookup_table = inputs
    embedding = embedding[1:]
    with tf.GradientTape() as tape:
        pred = model(embedding, lookup_table)
        cur_loss = loss_fn(pred, y_true)
        grads = tape.gradient(cur_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


def train():
    embedding, lookup_table, y_true = create_data()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, epsilon=1e-6)
    train_model(model, (embedding, lookup_table), y_true, optimizer)


if __name__ == "__main__":
    train()