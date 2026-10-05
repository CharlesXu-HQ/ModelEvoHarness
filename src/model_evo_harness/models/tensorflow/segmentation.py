"""TensorFlow soft-region mixture of local logistic regressions."""

import tensorflow as tf


class MLR(tf.keras.Model):
    """Returns a probability for dense [batch, feature] input."""

    def __init__(self, input_dim: int, num_regions: int):
        super().__init__()
        self.input_dim = input_dim
        self.region = tf.keras.layers.Dense(num_regions)
        self.local = tf.keras.layers.Dense(num_regions)

    def region_weights(self, x):
        return tf.nn.softmax(self.region(x), axis=-1)

    def call(self, x):
        return tf.reduce_sum(self.region_weights(x) * tf.nn.sigmoid(self.local(x)), axis=-1)
