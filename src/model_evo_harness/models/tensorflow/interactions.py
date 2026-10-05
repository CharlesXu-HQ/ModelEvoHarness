"""Independent TensorFlow cores for categorical feature interaction models.

Inputs are one integer ID per field, ``[batch, fields]``, except DCNv2, which
receives a preprocessed dense ``[batch, input_dim]`` vector. Outputs are logits.
"""

from itertools import combinations

import tensorflow as tf


class _Fields(tf.keras.layers.Layer):
    def __init__(self, cardinalities: list[int], embedding_dim: int):
        super().__init__()
        if len(cardinalities) < 2 or any(n <= 0 for n in cardinalities):
            raise ValueError("at least two positive field cardinalities are required")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.cardinalities = tuple(cardinalities)
        self.count = len(cardinalities)
        self.dim = embedding_dim
        self.pairs = tuple(combinations(range(self.count), 2))
        self.offsets = tf.constant([sum(cardinalities[:i]) for i in range(self.count)],
                                   dtype=tf.int32)
        total = sum(cardinalities)
        self.embeddings = tf.keras.layers.Embedding(total, embedding_dim)
        self.linear = tf.keras.layers.Embedding(total, 1)
        self.bias = self.add_weight(name="bias", shape=(), initializer="zeros")
        self.build((None, self.count))

    def build(self, input_shape):
        self.embeddings.build(input_shape)
        self.linear.build(input_shape)
        super().build(input_shape)

    def encode(self, fields: tf.Tensor) -> tf.Tensor:
        return self.embeddings(tf.cast(fields, tf.int32) + self.offsets)

    def first_order(self, fields: tf.Tensor) -> tf.Tensor:
        values = self.linear(tf.cast(fields, tf.int32) + self.offsets)
        return tf.squeeze(tf.reduce_sum(values, axis=1), axis=-1) + self.bias

    def pair_products(self, vectors: tf.Tensor) -> tf.Tensor:
        return tf.stack([vectors[:, i] * vectors[:, j] for i, j in self.pairs], axis=1)


class AFM(tf.keras.Model):
    """Attention-weighted pair products; categorical IDs ``[B,F]`` to logits ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 attention_dim: int = 16):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Dense(attention_dim, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        self.head = tf.keras.layers.Dense(1, use_bias=False)

    def attention_weights(self, fields: tf.Tensor) -> tf.Tensor:
        pairs = self.fields.pair_products(self.fields.encode(fields))
        return tf.nn.softmax(tf.squeeze(self.attention(pairs), axis=-1), axis=1)

    def call(self, fields: tf.Tensor) -> tf.Tensor:
        pairs = self.fields.pair_products(self.fields.encode(fields))
        weights = tf.nn.softmax(tf.squeeze(self.attention(pairs), axis=-1), axis=1)
        pooled = tf.reduce_sum(pairs * weights[:, :, None], axis=1)
        return self.fields.first_order(fields) + tf.squeeze(self.head(pooled), axis=-1)


class AutoInt(tf.keras.Model):
    """Field self-attention interactions; categorical IDs ``[B,F]`` to logits ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 num_heads: int = 2, num_layers: int = 2):
        super().__init__()
        if num_heads < 1 or embedding_dim % num_heads or num_layers < 1:
            raise ValueError("embedding_dim must divide num_heads and layers must be positive")
        self.fields = _Fields(cardinalities, embedding_dim)
        self.attentions = [tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim // num_heads,
            output_shape=embedding_dim) for _ in range(num_layers)]
        self.norms = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]
        self.head = tf.keras.layers.Dense(1)

    def interaction_features(self, fields: tf.Tensor) -> tf.Tensor:
        hidden = self.fields.encode(fields)
        for attention, norm in zip(self.attentions, self.norms):
            hidden = norm(hidden + attention(hidden, hidden))
        return hidden

    def call(self, fields: tf.Tensor) -> tf.Tensor:
        values = tf.reshape(self.interaction_features(fields),
                            (tf.shape(fields)[0], self.fields.count * self.fields.dim))
        return self.fields.first_order(fields) + tf.squeeze(self.head(values), axis=-1)


class FiBiNET(tf.keras.Model):
    """Field squeeze gates and bilinear pair terms; categorical IDs to ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 hidden_dim: int = 32):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        field_count = self.fields.count
        self.squeeze = tf.keras.Sequential([
            tf.keras.layers.Dense(max(1, field_count // 2), activation="relu"),
            tf.keras.layers.Dense(field_count, activation="sigmoid"),
        ])
        self.bilinear = tf.keras.layers.Dense(embedding_dim, use_bias=False)
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    def field_gates(self, fields: tf.Tensor) -> tf.Tensor:
        return self.squeeze(tf.reduce_mean(self.fields.encode(fields), axis=-1))

    def call(self, fields: tf.Tensor) -> tf.Tensor:
        vectors = self.fields.encode(fields)
        gated = vectors * self.squeeze(tf.reduce_mean(vectors, axis=-1))[:, :, None]
        products = [self.bilinear(vectors[:, i]) * vectors[:, j]
                    for i, j in self.fields.pairs]
        products += [self.bilinear(gated[:, i]) * gated[:, j]
                     for i, j in self.fields.pairs]
        return self.fields.first_order(fields) + tf.squeeze(
            self.head(tf.concat(products, axis=-1)), axis=-1)


class NFM(tf.keras.Model):
    """Bi-interaction pooling followed by an MLP; categorical IDs to ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 hidden_dim: int = 32):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        self.deep = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    @staticmethod
    def bi_interaction(vectors: tf.Tensor) -> tf.Tensor:
        return 0.5 * (tf.square(tf.reduce_sum(vectors, axis=1)) -
                      tf.reduce_sum(tf.square(vectors), axis=1))

    def call(self, fields: tf.Tensor) -> tf.Tensor:
        return self.fields.first_order(fields) + tf.squeeze(
            self.deep(self.bi_interaction(self.fields.encode(fields))), axis=-1)


class PNN(tf.keras.Model):
    """Inner and outer product layer before an MLP; categorical IDs to ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 hidden_dim: int = 32):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        self.deep = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    def product_features(self, vectors: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(vectors)[0]
        inner = tf.stack([tf.reduce_sum(vectors[:, i] * vectors[:, j], axis=-1)
                          for i, j in self.fields.pairs], axis=1)
        outer = tf.concat([tf.reshape(vectors[:, i, :, None] * vectors[:, j, None, :],
                                      (batch, self.fields.dim * self.fields.dim))
                           for i, j in self.fields.pairs], axis=1)
        flat = tf.reshape(vectors, (batch, self.fields.count * self.fields.dim))
        return tf.concat([flat, inner, outer], axis=1)

    def call(self, fields: tf.Tensor) -> tf.Tensor:
        return self.fields.first_order(fields) + tf.squeeze(
            self.deep(self.product_features(self.fields.encode(fields))), axis=-1)


class WideDeep(tf.keras.Model):
    """Explicit categorical crosses plus embedded MLP; IDs ``[B,F]`` to ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 hidden_dim: int = 32,
                 cross_pairs: list[tuple[int, int]] | None = None):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        self.cross_pairs = tuple(self.fields.pairs if cross_pairs is None else cross_pairs)
        if any(i < 0 or j >= self.fields.count or i >= j
               for i, j in self.cross_pairs):
            raise ValueError("cross_pairs must contain ordered field indices")
        self.cross_tables = [tf.keras.layers.Embedding(cardinalities[i] * cardinalities[j], 1)
                             for i, j in self.cross_pairs]
        self.deep = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    def cross_indices(self, fields: tf.Tensor) -> tf.Tensor:
        if not self.cross_pairs:
            return tf.zeros((tf.shape(fields)[0], 0), dtype=fields.dtype)
        return tf.stack([fields[:, i] * self.fields.cardinalities[j] + fields[:, j]
                         for i, j in self.cross_pairs], axis=1)

    def wide_logit(self, fields: tf.Tensor) -> tf.Tensor:
        indices = self.cross_indices(fields)
        cross = (tf.add_n([tf.squeeze(layer(indices[:, k]), axis=-1)
                           for k, layer in enumerate(self.cross_tables)])
                 if self.cross_tables else tf.zeros((tf.shape(fields)[0],)))
        return self.fields.first_order(fields) + cross

    def call(self, fields: tf.Tensor) -> tf.Tensor:
        return self.wide_logit(fields) + tf.squeeze(
            self.deep(self.fields.encode(fields)), axis=-1)


class XDeepFM(tf.keras.Model):
    """Compressed interaction network plus MLP; categorical IDs to ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 hidden_dim: int = 32, cin_channels: list[int] | None = None):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        channels = [8, 8] if cin_channels is None else cin_channels
        if not channels or any(n <= 0 for n in channels):
            raise ValueError("cin_channels must contain positive sizes")
        self.cin_weights = []
        previous = self.fields.count
        for i, width in enumerate(channels):
            self.cin_weights.append(self.add_weight(
                name=f"cin_weight_{i}", shape=(width, self.fields.count * previous),
                initializer="glorot_uniform"))
            previous = width
        self.deep = tf.keras.Sequential([
            tf.keras.layers.Flatten(), tf.keras.layers.Dense(hidden_dim, activation="relu"),
        ])
        self.head = tf.keras.layers.Dense(1)

    def cin_features(self, fields: tf.Tensor) -> tf.Tensor:
        original = self.fields.encode(fields)
        hidden = original
        outputs = []
        for weight in self.cin_weights:
            outer = original[:, :, None, :] * hidden[:, None, :, :]
            outer = tf.reshape(outer, (tf.shape(outer)[0], tf.shape(weight)[1],
                                       self.fields.dim))
            hidden = tf.nn.relu(tf.einsum("bmk,cm->bck", outer, weight))
            outputs.append(tf.reduce_sum(hidden, axis=-1))
        return tf.concat(outputs, axis=1)

    def call(self, fields: tf.Tensor) -> tf.Tensor:
        vectors = self.fields.encode(fields)
        features = tf.concat([self.cin_features(fields), self.deep(vectors)], axis=1)
        return self.fields.first_order(fields) + tf.squeeze(self.head(features), axis=-1)


class DCNv2(tf.keras.Model):
    """Matrix cross network with optional low rank plus MLP; dense ``[B,D]`` to ``[B]``."""

    def __init__(self, input_dim: int, num_cross_layers: int = 2,
                 hidden_dim: int = 32, rank: int | None = None):
        super().__init__()
        if input_dim <= 0 or num_cross_layers < 1 or (rank is not None and rank <= 0):
            raise ValueError("input_dim and cross layers must be positive; rank is positive")
        self.cross_layers = [tf.keras.Sequential([
            tf.keras.layers.Dense(rank, use_bias=False),
            tf.keras.layers.Dense(input_dim),
        ]) if rank is not None else tf.keras.layers.Dense(input_dim)
            for _ in range(num_cross_layers)]
        self.deep = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.head = tf.keras.layers.Dense(1)

    def cross_features(self, x: tf.Tensor) -> tf.Tensor:
        crossed = x
        for layer in self.cross_layers:
            crossed = crossed + x * layer(crossed)
        return crossed

    def call(self, x: tf.Tensor) -> tf.Tensor:
        features = tf.concat([self.cross_features(x), self.deep(x)], axis=1)
        return tf.squeeze(self.head(features), axis=-1)


class FwFM(tf.keras.Model):
    """Field-pair-weighted FM; categorical IDs ``[B,F]`` to logits ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        self.pair_weights = self.add_weight(name="pair_weights",
                                            shape=(len(self.fields.pairs),),
                                            initializer="ones")

    def first_order(self, fields: tf.Tensor) -> tf.Tensor:
        return self.fields.first_order(fields)

    def call(self, fields: tf.Tensor) -> tf.Tensor:
        vectors = self.fields.encode(fields)
        pairs = tf.stack([tf.reduce_sum(vectors[:, i] * vectors[:, j], axis=-1)
                          for i, j in self.fields.pairs], axis=1)
        return self.first_order(fields) + tf.reduce_sum(pairs * self.pair_weights, axis=1)
