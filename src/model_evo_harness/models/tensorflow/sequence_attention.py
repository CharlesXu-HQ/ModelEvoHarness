"""Native TensorFlow target pooling and padding-aware Dice activation.

Dice activation is distinct from the interest/conformity disentanglement DICE.
"""
import math

import tensorflow as tf


def _clean(values, valid):
    if valid.dtype != tf.bool or not values.dtype.is_floating:
        raise ValueError("floating values and boolean mask required")
    tf.debugging.assert_equal(tf.shape(valid), tf.shape(values)[:-1])
    values = tf.where(valid[..., None], values, tf.zeros_like(values))
    tf.debugging.assert_all_finite(values, "valid inputs must be finite")
    return values


class MaskedDice(tf.keras.layers.Layer):
    """Population moments over valid positions; frozen moving moments at eval.

    moving = momentum * moving + (1-momentum) * batch. Empty training batches
    do not update moments. No affine normalization; alpha learns the closed gate.
    """
    def __init__(self, dim, momentum=.9, epsilon=1e-5):
        super().__init__()
        if dim < 1 or not 0 <= momentum < 1 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("positive dim/epsilon and momentum in [0,1) required")
        self.dim, self.momentum, self.epsilon = dim, momentum, epsilon
        self.alpha = self.add_weight(name='alpha', shape=(dim,), initializer='zeros')
        self.moving_mean = self.add_weight(name='moving_mean', shape=(dim,), initializer='zeros', trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance', shape=(dim,), initializer='ones', trainable=False)

    def call(self, values, valid, training=None):
        values = _clean(values, valid)
        tf.debugging.assert_rank_at_least(values, 2)
        tf.debugging.assert_equal(tf.shape(values)[-1], self.dim)
        def batch_moments():
            selected = tf.boolean_mask(values, valid)
            mean = tf.reduce_mean(selected, 0)
            variance = tf.reduce_mean(tf.square(selected-mean), 0)
            self.moving_mean.assign(self.momentum*self.moving_mean + (1-self.momentum)*mean)
            self.moving_variance.assign(self.momentum*self.moving_variance + (1-self.momentum)*variance)
            return mean, variance
        use_batch = tf.logical_and(tf.cast(False if training is None else training, tf.bool), tf.reduce_any(valid))
        mean, variance = tf.cond(use_batch, batch_moments,
                                 lambda: (tf.identity(self.moving_mean), tf.identity(self.moving_variance)))
        gate = tf.sigmoid((values-mean)*tf.math.rsqrt(variance+self.epsilon))
        output = values*(gate+self.alpha*(1-gate))
        return tf.where(valid[..., None], output, tf.zeros_like(output))


class LocalActivationPooling(tf.keras.layers.Layer):
    """Target/event MLP with raw, sigmoid or masked-softmax temporal weights.

    Dict: representation [B,D], attention and masked scores [B,T]. Empty
    histories produce zero. Dice statistics exclude all masked positions.
    """
    def __init__(self, dim, hidden_dim, activation='relu', normalization='sigmoid'):
        super().__init__()
        if min(dim, hidden_dim) < 1 or activation not in ('relu', 'dice'):
            raise ValueError("positive sizes and relu/dice activation required")
        if normalization not in ('raw', 'sigmoid', 'softmax'):
            raise ValueError("normalization must be raw, sigmoid or softmax")
        self.dim, self.normalization = dim, normalization
        self.hidden = tf.keras.layers.Dense(hidden_dim)
        self.hidden.build((None, None, 4*dim))
        self.activation = MaskedDice(hidden_dim) if activation == 'dice' else tf.keras.layers.ReLU()
        self.score = tf.keras.layers.Dense(1)
        self.score.build((None, None, hidden_dim))

    def call(self, target, history, valid, training=None):
        tf.debugging.assert_rank(history, 3)
        tf.debugging.assert_equal(tf.shape(target), [tf.shape(history)[0], self.dim])
        tf.debugging.assert_equal(tf.shape(history)[-1], self.dim)
        tf.debugging.assert_all_finite(target, "target must be finite")
        history = _clean(history, valid)
        query = tf.broadcast_to(target[:, None], tf.shape(history))
        features = tf.concat([query, history, query-history, query*history], -1)
        hidden = self.hidden(features)
        hidden = self.activation(hidden, valid, training=training) if isinstance(self.activation, MaskedDice) else self.activation(hidden)
        scores = tf.squeeze(self.score(hidden), -1)
        if self.normalization == 'softmax':
            masked = tf.where(valid, scores, tf.cast(float('-inf'), scores.dtype))
            dummy = tf.where(tf.reduce_any(valid, -1, keepdims=True), tf.cast(float('-inf'), scores.dtype), tf.cast(0., scores.dtype))
            weights = tf.nn.softmax(tf.concat([masked, dummy], -1), -1)[:, :-1]
        else:
            weights = scores if self.normalization == 'raw' else tf.sigmoid(scores)
        weights = tf.where(valid, weights, tf.zeros_like(weights))
        return dict(representation=tf.reduce_sum(history*weights[..., None], 1), attention=weights,
                    scores=tf.where(valid, scores, tf.zeros_like(scores)))
