"""Independent Keras CTR tensor components, paired with the PyTorch formulas.

Reference: reczoo/FuxiCTR, commit 183da1c944ba8930fcdcd54114837c081959d748,
model_zoo/{GDCN,MaskNet,FinalMLP,EDCN,WuKong,FmFM,FLEN,FinalNet,QNN}.
Original implementations; host owns embeddings, head, objective and optimizer.
All field tensors must be fully defined (no padding). EDCN uses distinct
stream gates; QNN restores coordinate order after head operations.
"""
from itertools import combinations
from numbers import Integral
import math

import tensorflow as tf


def _positive(*values):
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError('dimensions and counts must be positive integers')


def _matrix(x, width):
    return tf.ensure_shape(x, (None, width))


def _tokens(x, fields, dim):
    return tf.ensure_shape(x, (None, fields, dim))


def _same_batch(x, y):
    tf.debugging.assert_equal(tf.shape(x)[0], tf.shape(y)[0], message='inputs must have the same batch size')


class GatedCross(tf.keras.layers.Layer):
    """[B,D] -> [B,D]; x += x0*(W(x)+b)*sigmoid(G(x)) at each depth."""
    def __init__(self, input_dim: int, num_layers: int = 2):
        super().__init__()
        _positive(input_dim, num_layers)
        self.input_dim = input_dim
        self.cross_layers = [tf.keras.layers.Dense(input_dim) for _ in range(num_layers)]
        self.gate_layers = [tf.keras.layers.Dense(input_dim, use_bias=False) for _ in range(num_layers)]

    def call(self, x):
        x = _matrix(x, self.input_dim)
        initial, hidden = x, x
        for cross, gate in zip(self.cross_layers, self.gate_layers):
            hidden = hidden + initial * cross(hidden) * tf.sigmoid(gate(hidden))
        return hidden


class MaskBlock(tf.keras.layers.Layer):
    """(original[B,I], hidden[B,H]) -> [B,O] with an unbounded MLP mask.

    ReLU(LN(W(mask_MLP(original)*hidden))). Compose with the same original
    input at serial depths or independent instances in parallel branches.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 mask_hidden_dim: int, layer_norm: bool = True):
        super().__init__()
        _positive(input_dim, hidden_dim, output_dim, mask_hidden_dim)
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.mask_in = tf.keras.layers.Dense(mask_hidden_dim)
        self.mask_out = tf.keras.layers.Dense(hidden_dim)
        self.projection = tf.keras.layers.Dense(output_dim, use_bias=False)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5) if layer_norm else tf.keras.layers.Identity()

    def call(self, original, hidden):
        original, hidden = _matrix(original, self.input_dim), _matrix(hidden, self.hidden_dim)
        _same_batch(original, hidden)
        mask = self.mask_out(tf.nn.relu(self.mask_in(original)))
        return tf.nn.relu(self.norm(self.projection(mask * hidden)))


class ContextFeatureGate(tf.keras.layers.Layer):
    """(features[B,F], context[B,C]) -> [B,F], using 2*sigmoid(MLP).

    Use independent instances for stream-specific gates; host supplies context.
    """
    def __init__(self, feature_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        _positive(feature_dim, context_dim, hidden_dim)
        self.feature_dim, self.context_dim = feature_dim, context_dim
        self.gate_in = tf.keras.layers.Dense(hidden_dim)
        self.gate_out = tf.keras.layers.Dense(feature_dim)

    def call(self, features, context):
        features, context = _matrix(features, self.feature_dim), _matrix(context, self.context_dim)
        _same_batch(features, context)
        return features * (2 * tf.sigmoid(self.gate_out(tf.nn.relu(self.gate_in(context)))))


class MultiHeadBilinearFusion(tf.keras.layers.Layer):
    """Two [B,D] streams -> [B,O]; affine + headwise bilinear terms.

    Contiguous stream partitions exclude cross-head products; no attention.
    """
    def __init__(self, left_dim: int, right_dim: int, output_dim: int = 1, num_heads: int = 1):
        super().__init__()
        _positive(left_dim, right_dim, output_dim, num_heads)
        if left_dim % num_heads or right_dim % num_heads:
            raise ValueError('stream dimensions must be divisible by num_heads')
        self.left_dim, self.right_dim, self.num_heads = left_dim, right_dim, num_heads
        self.left_linear = tf.keras.layers.Dense(output_dim)
        self.right_linear = tf.keras.layers.Dense(output_dim)
        self.interaction = self.add_weight(name='interaction', shape=(num_heads, left_dim // num_heads,
                                                                     right_dim // num_heads, output_dim), initializer='glorot_uniform')

    def call(self, left, right):
        left, right = _matrix(left, self.left_dim), _matrix(right, self.right_dim)
        _same_batch(left, right)
        x = tf.reshape(left, (tf.shape(left)[0], self.num_heads, self.left_dim // self.num_heads))
        y = tf.reshape(right, (tf.shape(right)[0], self.num_heads, self.right_dim // self.num_heads))
        return self.left_linear(left) + self.right_linear(right) + tf.einsum('bhi,hijo,bhj->bo', x, self.interaction, y)


class CrossDeepBridge(tf.keras.layers.Layer):
    """Two [B,D] streams -> [B,D] via product, sum, concat or attention.

    Attention has independent stream softmax gates, correcting source reuse.
    """
    def __init__(self, dim: int, mode: str = 'product'):
        super().__init__()
        _positive(dim)
        if mode not in ('product', 'sum', 'concat', 'attention'):
            raise ValueError('mode must be product, sum, concat or attention')
        self.dim, self.mode = dim, mode
        if mode == 'concat':
            self.projection = tf.keras.layers.Dense(dim)
        elif mode == 'attention':
            self.cross_gate = tf.keras.Sequential([tf.keras.layers.Dense(dim, activation='relu'), tf.keras.layers.Dense(dim)])
            self.deep_gate = tf.keras.Sequential([tf.keras.layers.Dense(dim, activation='relu'), tf.keras.layers.Dense(dim)])

    def call(self, cross, deep):
        cross, deep = _matrix(cross, self.dim), _matrix(deep, self.dim)
        _same_batch(cross, deep)
        if self.mode == 'product':
            return cross * deep
        if self.mode == 'sum':
            return cross + deep
        if self.mode == 'concat':
            return tf.nn.relu(self.projection(tf.concat((cross, deep), -1)))
        return cross * tf.nn.softmax(self.cross_gate(cross), -1) + deep * tf.nn.softmax(self.deep_gate(deep), -1)


class FieldRegulation(tf.keras.layers.Layer):
    """[B,F,D] -> two [B,F,D] tensors using global field softmax weights.

    Each weight is shared across embedding coordinates; no sample routing.
    """
    def __init__(self, num_fields: int, embedding_dim: int, temperature: float = 1.0):
        super().__init__()
        _positive(num_fields, embedding_dim)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        self.num_fields, self.embedding_dim, self.temperature = num_fields, embedding_dim, temperature
        self.cross_logits = self.add_weight(name='cross_logits', shape=(num_fields,), initializer='zeros')
        self.deep_logits = self.add_weight(name='deep_logits', shape=(num_fields,), initializer='zeros')

    def call(self, tokens):
        tokens = _tokens(tokens, self.num_fields, self.embedding_dim)
        cross = tf.nn.softmax(self.cross_logits / self.temperature)[None, :, None]
        deep = tf.nn.softmax(self.deep_logits / self.temperature)[None, :, None]
        return tokens * cross, tokens * deep


class WuKongBlock(tf.keras.layers.Layer):
    """[B,F,D] -> [B,interaction_tokens+linear_tokens,D].

    FMB: normalize X(X^T P), use a two-layer ReLU MLP to create tokens.
    Concatenate FMB and field-linear compression; add projected residual/LN.
    """
    def __init__(self, num_fields: int, embedding_dim: int, linear_tokens: int = 2,
                 interaction_tokens: int = 2, rank: int = 2, hidden_dim: int = 16,
                 layer_norm: bool = True):
        super().__init__()
        _positive(num_fields, embedding_dim, linear_tokens, interaction_tokens, rank, hidden_dim)
        self.num_fields, self.embedding_dim = num_fields, embedding_dim
        self.interaction_tokens = interaction_tokens
        self.output_tokens = linear_tokens + interaction_tokens
        self.gram_projection = self.add_weight(name='gram_projection', shape=(num_fields, rank), initializer='glorot_uniform')
        self.gram_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.fmb_in = tf.keras.layers.Dense(hidden_dim)
        self.fmb_out = tf.keras.layers.Dense(interaction_tokens * embedding_dim)
        self.linear_compression = tf.keras.layers.Dense(linear_tokens, use_bias=False)
        self.residual_projection = tf.keras.layers.Dense(self.output_tokens) if self.output_tokens != num_fields else tf.keras.layers.Identity()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5) if layer_norm else tf.keras.layers.Identity()
        self.rank = rank

    def compressed_gram(self, tokens):
        """[B,F,rank] affinities without materializing a full F*F Gram."""
        tokens = _tokens(tokens, self.num_fields, self.embedding_dim)
        return tf.matmul(tokens, tf.matmul(tf.transpose(tokens, (0, 2, 1)), self.gram_projection))

    def call(self, tokens):
        gram = self.gram_norm(tf.reshape(self.compressed_gram(tokens), (-1, self.num_fields * self.rank)))
        fmb = tf.nn.relu(self.fmb_out(tf.nn.relu(self.fmb_in(gram))))
        fmb = tf.reshape(fmb, (tf.shape(tokens)[0], self.interaction_tokens, self.embedding_dim))
        lcb = tf.transpose(self.linear_compression(tf.transpose(tokens, (0, 2, 1))), (0, 2, 1))
        residual = tf.transpose(self.residual_projection(tf.transpose(tokens, (0, 2, 1))), (0, 2, 1))
        return self.norm(tf.concat((fmb, lcb), 1) + residual)


class FieldMatrixInteraction(tf.keras.layers.Layer):
    """[B,F,D] -> lexicographic i<j pair scores [B,F*(F-1)/2].

    Matrix gives e_i^T M_ij e_j; vector gives sum(e_i*v_ij*e_j).
    """
    def __init__(self, num_fields: int, embedding_dim: int, interaction_type: str = 'matrix'):
        super().__init__()
        _positive(num_fields, embedding_dim)
        if num_fields < 2 or interaction_type not in ('matrix', 'vector'):
            raise ValueError('at least two fields and matrix/vector interaction_type required')
        self.num_fields, self.embedding_dim, self.interaction_type = num_fields, embedding_dim, interaction_type
        pairs = list(combinations(range(num_fields), 2))
        self.left_indices, self.right_indices = [i for i, _ in pairs], [j for _, j in pairs]
        shape = (len(pairs), embedding_dim, embedding_dim) if interaction_type == 'matrix' else (len(pairs), embedding_dim)
        self.pair_weight = self.add_weight(name='pair_weight', shape=shape, initializer='glorot_uniform')

    def call(self, tokens):
        tokens = _tokens(tokens, self.num_fields, self.embedding_dim)
        left, right = tf.gather(tokens, self.left_indices, axis=1), tf.gather(tokens, self.right_indices, axis=1)
        if self.interaction_type == 'matrix':
            return tf.einsum('bpi,pij,bpj->bp', left, self.pair_weight, right)
        return tf.reduce_sum(left * self.pair_weight * right, -1)


class GroupedFieldInteraction(tf.keras.layers.Layer):
    """[B,F,D] -> [B,D], weighted within-group FM + across-group products.

    Group IDs per field must be contiguous with at least two nonempty groups;
    generalizes FLEN's fixed user/item/context groups. Host supplies the head.
    """
    def __init__(self, field_groups, embedding_dim: int):
        super().__init__()
        _positive(embedding_dim)
        groups = tuple(field_groups)
        if not groups or any(not isinstance(g, Integral) or isinstance(g, bool) or g < 0 for g in groups):
            raise ValueError('field_groups must contain nonnegative integer IDs')
        count = max(groups) + 1
        if count < 2 or set(groups) != set(range(count)):
            raise ValueError('field_groups must cover contiguous IDs for at least two groups')
        self.num_fields, self.embedding_dim = len(groups), embedding_dim
        self.group_membership = tf.one_hot(groups, count)
        self.pairs = tuple(combinations(range(count), 2))
        self.within_weights = self.add_weight(name='within_weights', shape=(count,), initializer='ones')
        self.across_weights = self.add_weight(name='across_weights', shape=(len(self.pairs),), initializer='ones')

    def call(self, tokens):
        tokens = _tokens(tokens, self.num_fields, self.embedding_dim)
        membership = tf.cast(self.group_membership, tokens.dtype)
        pooled = tf.einsum('bfd,fg->bgd', tokens, membership)
        squared = tf.einsum('bfd,fg->bgd', tokens ** 2, membership)
        within = (pooled ** 2 - squared) * .5
        across = tf.stack([pooled[:, i] * pooled[:, j] for i, j in self.pairs], 1)
        return tf.reduce_sum(within * self.within_weights[None, :, None], 1) + tf.reduce_sum(across * self.across_weights[None, :, None], 1)


class FactorizedInteraction(tf.keras.layers.Layer):
    """[B,I] -> [B,O]; affine a,b give a+a*b or concat(a,a*b).

    Sum is also expressible by LatentCross + projection. No ensemble loss.
    """
    def __init__(self, input_dim: int, output_dim: int, residual_type: str = 'sum'):
        super().__init__()
        _positive(input_dim, output_dim)
        if residual_type not in ('sum', 'concat') or (residual_type == 'concat' and output_dim % 2):
            raise ValueError('residual_type must be sum/concat; concat needs an even output_dim')
        self.input_dim, self.residual_type = input_dim, residual_type
        self.projection = tf.keras.layers.Dense(2 * output_dim if residual_type == 'sum' else output_dim)

    def call(self, x):
        x = _matrix(x, self.input_dim)
        a, b = tf.split(self.projection(x), 2, -1)
        return a + a * b if self.residual_type == 'sum' else tf.concat((a, a * b), -1)


class QuadraticInteraction(tf.keras.layers.Layer):
    """[B,F*D] -> same shape; head/row QNN-alpha-style projected products.

    ReLU projected rows multiply matching field-coordinate heads; sum rows,
    restore original field-coordinate order, then add x. No dropout/SE loss.
    """
    def __init__(self, input_dim: int, num_fields: int, num_heads: int = 1, num_rows: int = 2):
        super().__init__()
        _positive(input_dim, num_fields, num_heads, num_rows)
        if input_dim % num_fields or (input_dim // num_fields) % num_heads:
            raise ValueError('input_dim must divide into fields with embedding_dim divisible by heads')
        self.input_dim, self.num_fields, self.num_heads, self.num_rows = input_dim, num_fields, num_heads, num_rows
        self.embedding_dim = input_dim // num_fields
        self.projection = tf.keras.layers.Dense(input_dim * num_rows)

    def call(self, x):
        x = _matrix(x, self.input_dim)
        batch, channels = tf.shape(x)[0], self.embedding_dim // self.num_heads
        heads = tf.transpose(tf.reshape(x, (batch, self.num_fields, self.num_heads, channels)), (0, 2, 1, 3))
        rows = tf.transpose(tf.reshape(tf.nn.relu(self.projection(x)), (batch, self.num_rows, self.num_fields, self.num_heads, channels)), (0, 3, 1, 2, 4))
        products = tf.reduce_sum(heads[:, :, None] * rows, 2)
        return x + tf.reshape(tf.transpose(products, (0, 2, 1, 3)), (batch, self.input_dim))
