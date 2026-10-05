"""Small, independently written TensorFlow recommendation model references."""

import tensorflow as tf


class FM(tf.keras.Model):
    """Second-order factorization machine for one categorical ID per field."""

    def __init__(self, cardinalities: list[int], embedding_dim: int):
        super().__init__()
        self.linear = [tf.keras.layers.Embedding(n, 1) for n in cardinalities]
        self.embeddings = [tf.keras.layers.Embedding(n, embedding_dim)
                           for n in cardinalities]
        self.bias = self.add_weight(name="bias", shape=(), initializer="zeros")

    def _parts(self, fields: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        linear = tf.add_n([tf.squeeze(layer(fields[:, i]), axis=-1)
                           for i, layer in enumerate(self.linear)]) + self.bias
        vectors = tf.stack([layer(fields[:, i])
                            for i, layer in enumerate(self.embeddings)], axis=1)
        summed = tf.reduce_sum(vectors, axis=1)
        interaction = 0.5 * tf.reduce_sum(
            tf.square(summed) - tf.reduce_sum(tf.square(vectors), axis=1), axis=1)
        return linear + interaction, vectors

    def call(self, fields: tf.Tensor) -> tf.Tensor:
        return self._parts(fields)[0]


class DeepFM(FM):
    """FM and MLP trained jointly from the same field embeddings."""

    def __init__(self, cardinalities: list[int], embedding_dim: int, hidden_dim: int):
        super().__init__(cardinalities, embedding_dim)
        self.deep = tf.keras.Sequential([
            tf.keras.layers.Flatten(), tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, fields: tf.Tensor) -> tf.Tensor:
        fm_logit, vectors = self._parts(fields)
        return fm_logit + tf.squeeze(self.deep(vectors), axis=-1)


class DCN(tf.keras.Model):
    """Original vector-cross DCN with a parallel MLP branch."""

    def __init__(self, input_dim: int, num_cross_layers: int, hidden_dim: int):
        super().__init__()
        self.cross_weights = [self.add_weight(name=f"cross_weight_{i}",
                                              shape=(input_dim, 1),
                                              initializer="glorot_uniform")
                              for i in range(num_cross_layers)]
        self.cross_biases = [self.add_weight(name=f"cross_bias_{i}",
                                             shape=(input_dim,), initializer="zeros")
                             for i in range(num_cross_layers)]
        self.deep = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.head = tf.keras.layers.Dense(1)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        crossed = x
        for weight, bias in zip(self.cross_weights, self.cross_biases):
            crossed = crossed + x * tf.matmul(crossed, weight) + bias
        return tf.squeeze(self.head(tf.concat([crossed, self.deep(x)], axis=-1)), axis=-1)


class DIN(tf.keras.Model):
    """Target-conditioned attention over a supplied history embedding sequence."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, target: tf.Tensor, history: tf.Tensor,
             mask: tf.Tensor) -> tf.Tensor:
        repeated = tf.broadcast_to(target[:, None, :], tf.shape(history))
        attention_input = tf.concat([repeated, history, repeated - history,
                                     repeated * history], axis=-1)
        scores = tf.squeeze(self.attention(attention_input), axis=-1)
        # DIN retains activation intensity instead of normalizing weights to sum to one.
        weights = tf.sigmoid(scores) * tf.cast(mask, scores.dtype)
        pooled = tf.reduce_sum(history * weights[:, :, None], axis=1)
        return tf.squeeze(self.head(tf.concat([target, pooled], axis=-1)), axis=-1)


class MMoE(tf.keras.Model):
    """Multiple shared experts with an independent gate and head per task."""

    def __init__(self, input_dim: int, num_tasks: int, num_experts: int,
                 expert_dim: int):
        super().__init__()
        self.experts = [tf.keras.layers.Dense(expert_dim, activation="relu")
                        for _ in range(num_experts)]
        self.gates = [tf.keras.layers.Dense(num_experts) for _ in range(num_tasks)]
        self.heads = [tf.keras.layers.Dense(1) for _ in range(num_tasks)]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        expert_values = tf.stack([expert(x) for expert in self.experts], axis=1)
        outputs = []
        for gate, head in zip(self.gates, self.heads):
            weights = tf.nn.softmax(gate(x), axis=1)[:, :, None]
            outputs.append(head(tf.reduce_sum(expert_values * weights, axis=1)))
        return tf.concat(outputs, axis=1)


class TwoTower(tf.keras.Model):
    """Independent user/item encoders with a batchwise dot-product score matrix."""

    def __init__(self, user_dim: int, item_dim: int, embedding_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ])
        self.item_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ])

    def encode_user(self, user_features: tf.Tensor) -> tf.Tensor:
        return self.user_tower(user_features)

    def encode_item(self, item_features: tf.Tensor) -> tf.Tensor:
        return self.item_tower(item_features)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        user_features, item_features = inputs
        return tf.matmul(self.encode_user(user_features),
                         self.encode_item(item_features), transpose_b=True)
