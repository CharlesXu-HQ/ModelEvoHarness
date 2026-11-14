"""Original native Keras APG linear adapter and PPNet prior gate.

Reference: reczoo/FuxiCTR at 183da1c944ba8930fcdcd54114837c081959d748,
model_zoo/APG/src/APG.py and model_zoo/PEPNet/src/PPNet.py. Standalone
hidden-state components; host owns embeddings, head and training objective.
"""
from numbers import Integral

import tensorflow as tf


def _positive(*values):
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError('dimensions must be positive integers')


class AdaptiveLinear(tf.keras.layers.Layer):
    """(features[B,I], condition[B,C]) -> [B,O]. Full x@W(c) or x@U@S(c)@V.

    U/V are shared and optionally overparameterized matrix products. Generated
    bias follows flattened W/S in the affine generator; otherwise it is shared.
    The host passes layer(x,x) for explicit self-wise conditioning.
    """
    def __init__(self, input_dim: int, output_dim: int, condition_dim: int,
                 rank: int | None = None, generated_bias: bool = False,
                 overparam_dim: int | None = None):
        super().__init__()
        _positive(input_dim, output_dim, condition_dim)
        if rank is not None:
            _positive(rank)
            if rank > min(input_dim, output_dim):
                raise ValueError('rank cannot exceed input or output dimensions')
        if overparam_dim is not None:
            _positive(overparam_dim)
            if rank is None or overparam_dim < rank:
                raise ValueError('overparam_dim requires a rank and must be at least that rank')
        self.input_dim, self.output_dim, self.condition_dim = input_dim, output_dim, condition_dim
        self.rank, self.generated_bias, self.overparam_dim = rank, generated_bias, overparam_dim
        self.matrix_size = rank * rank if rank is not None else input_dim * output_dim
        self.generator = tf.keras.layers.Dense(self.matrix_size + (output_dim if generated_bias else 0))
        if rank is not None:
            if overparam_dim is None:
                self.input_factor = self.add_weight(name='input_factor', shape=(input_dim, rank), initializer='glorot_uniform')
                self.output_factor = self.add_weight(name='output_factor', shape=(rank, output_dim), initializer='glorot_uniform')
            else:
                self.input_left = self.add_weight(name='input_left', shape=(input_dim, overparam_dim), initializer='glorot_uniform')
                self.input_right = self.add_weight(name='input_right', shape=(overparam_dim, rank), initializer='glorot_uniform')
                self.output_left = self.add_weight(name='output_left', shape=(rank, overparam_dim), initializer='glorot_uniform')
                self.output_right = self.add_weight(name='output_right', shape=(overparam_dim, output_dim), initializer='glorot_uniform')
        self.bias = None if generated_bias else self.add_weight(name='bias', shape=(output_dim,), initializer='zeros')

    def call(self, features, condition):
        features = tf.ensure_shape(features, (None, self.input_dim))
        condition = tf.ensure_shape(condition, (None, self.condition_dim))
        tf.debugging.assert_equal(tf.shape(features)[0], tf.shape(condition)[0], message='features and condition must have the same batch size')
        generated = self.generator(condition)
        bias = generated[:, self.matrix_size:] if self.generated_bias else self.bias
        if self.rank is None:
            matrix = tf.reshape(generated[:, :self.matrix_size], (tf.shape(features)[0], self.input_dim, self.output_dim))
            output = tf.squeeze(tf.matmul(features[:, None], matrix), 1)
        else:
            u = self.input_factor if self.overparam_dim is None else tf.matmul(self.input_left, self.input_right)
            v = self.output_factor if self.overparam_dim is None else tf.matmul(self.output_left, self.output_right)
            center = tf.reshape(generated[:, :self.matrix_size], (tf.shape(features)[0], self.rank, self.rank))
            output = tf.matmul(tf.squeeze(tf.matmul(tf.matmul(features, u)[:, None], center), 1), v)
        return output + bias


class PriorConditionedGate(tf.keras.layers.Layer):
    """(hidden[B,H], features[B,F], prior[B,P]) -> [B,H].

    hidden * 2*sigmoid(MLP([stopgrad(features), prior])). detach_features is
    True by default; hidden/prior remain differentiable. Compose per depth.
    """
    def __init__(self, feature_dim: int, prior_dim: int, hidden_dim: int,
                 gate_hidden_dim: int, detach_features: bool = True):
        super().__init__()
        _positive(feature_dim, prior_dim, hidden_dim, gate_hidden_dim)
        self.feature_dim, self.prior_dim, self.hidden_dim = feature_dim, prior_dim, hidden_dim
        self.detach_features = detach_features
        self.gate_in = tf.keras.layers.Dense(gate_hidden_dim)
        self.gate_out = tf.keras.layers.Dense(hidden_dim)

    def call(self, hidden, features, prior):
        hidden = tf.ensure_shape(hidden, (None, self.hidden_dim))
        features = tf.ensure_shape(features, (None, self.feature_dim))
        prior = tf.ensure_shape(prior, (None, self.prior_dim))
        tf.debugging.assert_equal(tf.shape(hidden)[0], tf.shape(features)[0], message='batch size mismatch')
        tf.debugging.assert_equal(tf.shape(hidden)[0], tf.shape(prior)[0], message='batch size mismatch')
        base = tf.stop_gradient(features) if self.detach_features else features
        gate = self.gate_out(tf.nn.relu(self.gate_in(tf.concat((base, prior), -1))))
        return hidden * (2 * tf.sigmoid(gate))
