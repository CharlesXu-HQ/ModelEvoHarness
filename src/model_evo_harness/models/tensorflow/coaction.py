"""Independent native Keras CAN micro-MLP, without host embedding tables.

Reference: CAN (arXiv:2011.05625), section 4; author implementation
CAN-Paper/Co-Action-Network at 98a7219105568b9420cf28467877d08a1c56ae69.
Order summation (paper) and concatenation (author code) are explicit options.
No third-party model code is imported.
"""
from numbers import Integral

import tensorflow as tf


class CoAction(tf.keras.layers.Layer):
    """(parameters[B,P], feed[B,T,D], valid[B,T]) -> [B,T,C].

    Per sample, parameters contains row-major W_1[D,d1], optional b_1[d1],
    W_2[d1,d2], optional b_2[d2], and so on, in layer order. Thus
    P=sum(d[l-1]*d[l] + (d[l] if use_bias else 0)).

    For q=1..order, start at feed**q and apply the same per-sample matrices
    and biases. Each layer computes h@W+b; tanh applies to nonfinal layers.
    Readouts concatenate all layer outputs. order_reduction='concat' joins
    readouts in ascending power order; 'sum' adds them elementwise. C is
    sum(layer_dims)*order for concat, otherwise sum(layer_dims).

    True means valid. Padding is zeroed before powers and after readout;
    all-invalid samples ignore their parameters. parameters/feed must share
    a floating dtype; valid must be boolean. Host-owned tables supply the
    trainable inputs. This layer owns no weights and does not pool history.
    """
    def __init__(self, input_dim: int, layer_dims, order: int = 1,
                 order_reduction: str = 'concat', use_bias: bool = False):
        super().__init__(autocast=False)
        try:
            layer_dims = tuple(layer_dims)
        except TypeError as exc:
            raise ValueError('layer_dims must contain positive integer dimensions') from exc
        if not layer_dims or any(not isinstance(v, Integral) or isinstance(v, bool) or v < 1
                                 for v in (input_dim, order, *layer_dims)):
            raise ValueError('input_dim, order and layer_dims must be positive integers')
        if order_reduction not in ('concat', 'sum'):
            raise ValueError('order_reduction must be concat or sum')
        if not isinstance(use_bias, bool):
            raise ValueError('use_bias must be boolean')
        self.input_dim, self.layer_dims = input_dim, layer_dims
        self.order, self.order_reduction, self.use_bias = order, order_reduction, use_bias
        widths = (input_dim, *layer_dims)
        self.parameter_dim = sum(a*b + (b if use_bias else 0) for a, b in zip(widths, widths[1:]))
        self.output_dim = sum(layer_dims) * (order if order_reduction == 'concat' else 1)

    def call(self, parameters, feed, valid):
        if not parameters.dtype.is_floating or not feed.dtype.is_floating or parameters.dtype != feed.dtype:
            raise TypeError('parameters and feed must have the same floating dtype')
        if valid.dtype != tf.bool:
            raise TypeError('valid must be boolean')
        parameters = tf.ensure_shape(parameters, (None, self.parameter_dim))
        feed = tf.ensure_shape(feed, (None, None, self.input_dim))
        valid = tf.ensure_shape(valid, (None, None))
        tf.debugging.assert_equal(tf.shape(parameters)[0], tf.shape(feed)[0], message='batch size mismatch')
        tf.debugging.assert_equal(tf.shape(valid), tf.shape(feed)[:2], message='valid must match feed [B,T]')
        feed = tf.where(valid[..., None], feed, tf.zeros_like(feed))
        parameters = tf.where(tf.reduce_any(valid, -1, keepdims=True), parameters, tf.zeros_like(parameters))
        layers, offset, width = [], 0, self.input_dim
        for size in self.layer_dims:
            count = width * size
            matrix = tf.reshape(parameters[:, offset:offset+count], (tf.shape(parameters)[0], width, size))
            offset += count
            bias = parameters[:, offset:offset+size] if self.use_bias else None
            offset += size if self.use_bias else 0
            layers.append((matrix, bias))
            width = size
        orders = []
        for power in range(1, self.order + 1):
            hidden, outputs = tf.pow(feed, power), []
            for index, (matrix, bias) in enumerate(layers):
                hidden = tf.matmul(hidden, matrix)
                if bias is not None:
                    hidden = hidden + bias[:, None]
                if index < len(layers) - 1:
                    hidden = tf.tanh(hidden)
                outputs.append(hidden)
            orders.append(tf.concat(outputs, axis=-1))
        output = tf.concat(orders, axis=-1) if self.order_reduction == 'concat' else tf.add_n(orders)
        return tf.where(valid[..., None], output, tf.zeros_like(output))
