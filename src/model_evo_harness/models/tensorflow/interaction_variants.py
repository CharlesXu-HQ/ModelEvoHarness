"""Native TensorFlow cross-mixture and input-aware FM blocks.

Independent DCN-Mix, IFM and DIFM mechanisms, with no external model framework.
Hosts own embeddings, task heads, global bias, losses and optimization.
"""

from numbers import Integral

import tensorflow as tf


def _positive(*values):
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError('dimensions must be positive integers')


def _shape(x, dimensions):
    if not x.dtype.is_floating:
        raise ValueError('inputs must be floating tensors')
    x = tf.ensure_shape(x, (None, *dimensions))
    return tf.debugging.check_numerics(x, 'inputs must be finite; sanitize padding before this block')


class CrossNetMix(tf.keras.layers.Layer):
    """Dense [B,D] -> [B,D], U tanh(C tanh(V^T h)) experts mixed by softmax.

    Layer update is h+x0*(sum_e gate_e*expert_e+b_l). Each layer has distinct
    U/V/C; the bias-free router is shared across layers. return_routing=True
    also returns probabilities [B,L,E]. No deep branch or prediction head.
    Rank/expert changes alter capacity and cost; no efficiency claim is made.
    """
    def __init__(self, input_dim, rank=8, num_experts=4, num_layers=2):
        super().__init__()
        _positive(input_dim, rank, num_experts, num_layers)
        self.input_dim, self.num_layers = input_dim, num_layers
        # Per-matrix fan-in/fan-out scaling, independent of expert/layer count.
        uv_init = tf.keras.initializers.RandomUniform(-(6 / (input_dim + rank)) ** .5, (6 / (input_dim + rank)) ** .5)
        c_init = tf.keras.initializers.RandomUniform(-(3 / rank) ** .5, (3 / rank) ** .5)
        self.u = self.add_weight(name='u', shape=(num_layers, num_experts, input_dim, rank), initializer=uv_init)
        self.v = self.add_weight(name='v', shape=(num_layers, num_experts, input_dim, rank), initializer=tf.keras.initializers.RandomUniform(-(6 / (input_dim + rank)) ** .5, (6 / (input_dim + rank)) ** .5))
        self.c = self.add_weight(name='c', shape=(num_layers, num_experts, rank, rank), initializer=c_init)
        self.router = self.add_weight(name='router', shape=(input_dim, num_experts), initializer='glorot_uniform')
        self.bias = self.add_weight(name='bias', shape=(num_layers, input_dim), initializer='zeros')

    def call(self, x, return_routing=False):
        x = _shape(x, (self.input_dim,))
        hidden, routing = x, []
        for index in range(self.num_layers):
            weights = tf.nn.softmax(hidden @ self.router, axis=-1)
            low = tf.tanh(tf.einsum('bd,edr->ber', hidden, self.v[index]))
            low = tf.tanh(tf.einsum('ers,bes->ber', self.c[index], low))
            experts = tf.einsum('edr,ber->bed', self.u[index], low)
            hidden = hidden + x * (tf.reduce_sum(experts * weights[..., None], axis=1) + self.bias[index])
            routing.append(weights)
        return (hidden, tf.stack(routing, 1)) if return_routing else hidden


class InputAwareFM(tf.keras.layers.Layer):
    """Embeddings [B,F,D], first-order terms [B,F] -> jointly weighted logit [B].

    sum_i(a_i*w_i)+sum_{i<j}<a_i*e_i,a_j*e_j>; host owns global bias.
    IFM factors are F*softmax(MLP(flat(e)) @ bit_output). DIFM adds an MLP
    field-factor path to a scaled multi-head field-attention factor path,
    with projected residual and ReLU, without a final softmax/positivity.
    factor_components returns both unnormalized paths; field_factors returns
    the final [B,F] factors. All field slots must be observed and finite.
    """
    def __init__(self, num_fields, embedding_dim, hidden_dims=(32,), mode='ifm', num_heads=2):
        super().__init__()
        _positive(num_fields, embedding_dim, *hidden_dims)
        if num_fields < 2 or mode not in ('ifm', 'difm'):
            raise ValueError('at least two fields and ifm/difm mode required')
        if mode == 'difm':
            _positive(num_heads)
            if embedding_dim % num_heads:
                raise ValueError('embedding_dim must be divisible by num_heads')
        self.num_fields, self.embedding_dim, self.mode = num_fields, embedding_dim, mode
        self.num_heads = num_heads
        self.bit_mlp = [tf.keras.layers.Dense(size, activation='relu') for size in hidden_dims]
        width = hidden_dims[-1] if hidden_dims else num_fields * embedding_dim
        self.bit_output = self.add_weight(name='bit_output', shape=(width, num_fields), initializer='glorot_uniform')
        if mode == 'difm':
            for name in ('query', 'key', 'value', 'residual'):
                setattr(self, name, self.add_weight(name=name, shape=(embedding_dim, embedding_dim), initializer='glorot_uniform'))
            self.vector_output = self.add_weight(name='vector_output', shape=(num_fields * embedding_dim, num_fields), initializer='glorot_uniform')

    def factor_components(self, tokens):
        """Return unnormalized (bit, vector) [B,F] factors; vector=0 for IFM."""
        tokens = _shape(tokens, (self.num_fields, self.embedding_dim))
        batch = tf.shape(tokens)[0]
        bit = tf.reshape(tokens, (batch, self.num_fields * self.embedding_dim))
        for dense in self.bit_mlp:
            bit = dense(bit)
        bit = bit @ self.bit_output
        if self.mode == 'ifm':
            return bit, tf.zeros_like(bit)
        width = self.embedding_dim // self.num_heads
        q, k, v = [tf.transpose(tf.reshape(tokens @ matrix, (batch, self.num_fields, self.num_heads, width)), (0, 2, 1, 3))
                   for matrix in (self.query, self.key, self.value)]
        weights = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / width ** .5, axis=-1)
        attended = tf.reshape(tf.transpose(weights @ v, (0, 2, 1, 3)), (batch, self.num_fields, self.embedding_dim))
        attended = tf.nn.relu(attended + tokens @ self.residual)
        return bit, tf.reshape(attended, (batch, self.num_fields * self.embedding_dim)) @ self.vector_output

    def field_factors(self, tokens):
        bit, vector = self.factor_components(tokens)
        return self.num_fields * tf.nn.softmax(bit, axis=-1) if self.mode == 'ifm' else bit + vector

    def call(self, tokens, first_order_fields):
        first_order_fields = _shape(first_order_fields, (self.num_fields,))
        tf.debugging.assert_equal(tf.shape(tokens)[0], tf.shape(first_order_fields)[0])
        factors = self.field_factors(tokens)
        weighted = tokens * factors[..., None]
        pair_sum = .5 * tf.reduce_sum(tf.square(tf.reduce_sum(weighted, 1)) - tf.reduce_sum(tf.square(weighted), 1), -1)
        return tf.reduce_sum(first_order_fields * factors, -1) + pair_sum


class DualInputAwareFM(InputAwareFM):
    """DIFM entry point; same input/output contract as InputAwareFM."""
    def __init__(self, num_fields, embedding_dim, hidden_dims=(32,), num_heads=2):
        super().__init__(num_fields, embedding_dim, hidden_dims, mode='difm', num_heads=num_heads)
