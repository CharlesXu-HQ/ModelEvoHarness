"""TensorFlow ranking tensor cores; feature pipelines and indexes are caller-owned."""

import tensorflow as tf


class DeepCrossing(tf.keras.Model):
    """Residual feature crossing; ``features [B,D] -> logits [B]``."""

    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int = 2):
        super().__init__()
        self.stack = tf.keras.layers.Dense(hidden_dim)
        self.blocks = [tf.keras.Sequential((tf.keras.layers.Dense(hidden_dim, activation="relu"),
                                            tf.keras.layers.Dense(hidden_dim)))
                       for _ in range(num_blocks)]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, features: tf.Tensor) -> tf.Tensor:
        x = self.stack(features)
        for block in self.blocks:
            x = tf.nn.relu(x + block(x))
        return tf.squeeze(self.output_layer(x), -1)


class FNN(tf.keras.Model):
    """FM-initializable field embeddings and MLP; does not train the FM."""

    def __init__(self, field_cardinalities: tuple[int, ...], embedding_dim: int,
                 hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        self.embeddings = [tf.keras.layers.Embedding(n, embedding_dim)
                           for n in field_cardinalities]
        self.mlp = tf.keras.Sequential([*(tf.keras.layers.Dense(n, activation="relu")
                                         for n in hidden_dims), tf.keras.layers.Dense(1)])

    def load_fm_embeddings(self, weights: list[tf.Tensor]) -> None:
        if len(weights) != len(self.embeddings):
            raise ValueError("one FM embedding matrix is required per field")
        for layer, weight in zip(self.embeddings, weights):
            if tuple(weight.shape) != (layer.input_dim, layer.output_dim):
                raise ValueError("FM embedding shape mismatch")
            layer.build((None,))
            layer.embeddings.assign(weight)

    def call(self, field_ids: tf.Tensor) -> tf.Tensor:
        if field_ids.shape[-1] != len(self.embeddings):
            raise ValueError("field count mismatch")
        x = tf.concat([emb(field_ids[:, i]) for i, emb in enumerate(self.embeddings)], -1)
        return tf.squeeze(self.mlp(x), -1)


class NeuMF(tf.keras.Model):
    """Independent GMF and MLP ID embeddings; IDs [B] -> logits [B]."""

    def __init__(self, num_users: int, num_items: int, dim: int,
                 mlp_dims: tuple[int, ...] = (32, 16)):
        super().__init__()
        self.user_gmf = tf.keras.layers.Embedding(num_users, dim)
        self.item_gmf = tf.keras.layers.Embedding(num_items, dim)
        self.user_mlp = tf.keras.layers.Embedding(num_users, dim)
        self.item_mlp = tf.keras.layers.Embedding(num_items, dim)
        self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(n, activation="relu")
                                         for n in mlp_dims])
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, user_ids: tf.Tensor, item_ids: tf.Tensor) -> tf.Tensor:
        gmf = self.user_gmf(user_ids) * self.item_gmf(item_ids)
        deep = self.mlp(tf.concat((self.user_mlp(user_ids), self.item_mlp(item_ids)), -1))
        return tf.squeeze(self.output_layer(tf.concat((gmf, deep), -1)), -1)


class LatentCross(tf.keras.Model):
    """Context multiplication on hidden states [B,T,D] or [B,D]."""

    def __init__(self, hidden_dim: int, context_dim: int):
        super().__init__()
        self.context = tf.keras.layers.Dense(hidden_dim)

    def call(self, hidden: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        gate = self.context(context)
        if hidden.shape.rank == 3 and gate.shape.rank == 2:
            gate = gate[:, None, :]
        return hidden * (1 + gate)


class MultiViewDSSM(tf.keras.Model):
    """Shared user and domain-specific item encoders in a cosine space."""

    def __init__(self, user_dim: int, item_dims: tuple[int, ...], embedding_dim: int):
        super().__init__()
        self.user_tower = tf.keras.layers.Dense(embedding_dim)
        self.item_towers = [tf.keras.layers.Dense(embedding_dim) for _ in item_dims]
        self.user_tower.build((None, user_dim))
        for tower, item_dim in zip(self.item_towers, item_dims):
            tower.build((None, item_dim))

    def encode_user(self, features: tf.Tensor) -> tf.Tensor:
        return tf.math.l2_normalize(self.user_tower(features), -1)

    def encode_item(self, features: tf.Tensor, domain: int) -> tf.Tensor:
        return tf.math.l2_normalize(self.item_towers[domain](features), -1)

    def call(self, user: tf.Tensor, item: tf.Tensor, domain: int) -> tf.Tensor:
        return tf.reduce_sum(self.encode_user(user) * self.encode_item(item, domain), -1)


class PredictiveBilinear(tf.keras.Model):
    """Feature-based score ``user[B,U] @ W[U,I] · item[B,I]``."""

    def __init__(self, user_dim: int, item_dim: int):
        super().__init__()
        self.weight = self.add_weight(name="bilinear", shape=(user_dim, item_dim),
                                      initializer="glorot_uniform")

    def call(self, user: tf.Tensor, item: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bu,ui,bi->b", user, self.weight, item)


class DPPGreedy(tf.keras.Model):
    """Fixed-capacity greedy determinant selection; PSD kernel -> -1-padded [B,K].

    Stop when no eligible conditional variance exceeds tolerance, rather than
    when log-det gain is negative. This is not unconstrained-cardinality MAP.
    """

    def __init__(self, slate_size: int, tolerance: float = 1e-10):
        super().__init__()
        self.slate_size = slate_size
        self.tolerance = tolerance

    def call(self, kernel: tf.Tensor, eligibility: tf.Tensor) -> tf.Tensor:
        if kernel.shape.rank != 3 or kernel.shape[-1] != kernel.shape[-2]:
            raise ValueError("kernel must be [B,N,N]")
        tf.debugging.assert_equal(tf.shape(kernel)[1], tf.shape(kernel)[2])
        tf.debugging.assert_equal(tf.shape(eligibility), tf.shape(kernel)[:2])

        def select(inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
            matrix, available = inputs
            matrix = tf.stop_gradient(matrix)
            if matrix.dtype in (tf.float16, tf.bfloat16):
                matrix = tf.cast(matrix, tf.float32)
            n = tf.shape(matrix)[0]
            c = tf.zeros((n, self.slate_size), matrix.dtype)
            residual = tf.linalg.diag_part(matrix)
            result = tf.fill((self.slate_size,), -1)

            def cond(step, factors, gains, allowed, chosen):
                return ((step < self.slate_size) &
                        tf.reduce_any(allowed & (gains > self.tolerance)))

            def body(step, factors, gains, allowed, chosen):
                index = tf.argmax(tf.where(allowed, gains,
                                           tf.cast(float("-inf"), gains.dtype)),
                                  output_type=tf.int32)
                chosen = tf.tensor_scatter_nd_update(chosen, [[step]], [index])
                old_row = tf.gather(factors, index)
                update = (tf.gather(matrix, index) -
                          tf.linalg.matvec(factors, old_row)) / tf.sqrt(tf.gather(gains, index))
                factors += update[:, None] * tf.one_hot(step, self.slate_size,
                                                        dtype=matrix.dtype)[None]
                gains -= tf.square(update)
                allowed &= ~tf.one_hot(index, n, on_value=True, off_value=False,
                                        dtype=tf.bool)
                return step + 1, factors, gains, allowed, chosen

            return tf.while_loop(cond, body, (0, c, residual, available, result))[4]

        return tf.map_fn(select, (kernel, eligibility),
                         fn_output_signature=tf.TensorSpec((self.slate_size,), tf.int32))


class RankNet(tf.keras.Model):
    """Mean RankNet pair loss within each query row; ties/padding excluded."""

    def call(self, scores: tf.Tensor, labels: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(mask, tf.bool)
        scores = tf.where(mask, scores, tf.zeros_like(scores))
        labels = tf.where(mask, tf.cast(labels, scores.dtype), tf.zeros_like(scores))
        pairs = (labels[:, :, None] > labels[:, None, :]) & mask[:, :, None] & mask[:, None, :]
        loss = tf.nn.softplus(-(scores[:, :, None] - scores[:, None, :]))
        weight = tf.cast(pairs, loss.dtype)
        return tf.reduce_sum(loss * weight) / tf.maximum(tf.reduce_sum(weight), 1)


class LambdaRank(tf.keras.Model):
    """RankNet pair loss weighted by stop-gradient swap DeltaNDCG per query."""

    def call(self, scores: tf.Tensor, labels: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(mask, tf.bool)
        scores = tf.where(mask, scores, tf.zeros_like(scores))
        labels = tf.where(mask, tf.cast(labels, scores.dtype), tf.zeros_like(scores))
        n = tf.shape(scores)[1]
        pairs = (labels[:, :, None] > labels[:, None, :]) & mask[:, :, None] & mask[:, None, :]
        order = tf.argsort(tf.where(mask, scores, tf.cast(float("-inf"), scores.dtype)),
                           axis=-1, direction="DESCENDING")
        position = tf.argsort(order, axis=-1)
        discount = 1 / (tf.math.log(tf.cast(position + 2, scores.dtype)) / tf.math.log(2.0))
        gains = tf.pow(2.0, labels) - 1
        ideal = tf.sort(tf.where(mask, labels, tf.cast(float("-inf"), labels.dtype)),
                        axis=-1, direction="DESCENDING")
        ideal_gain = tf.where(tf.math.is_finite(ideal), tf.pow(2.0, ideal) - 1, 0)
        ideal_discount = 1 / (tf.math.log(tf.cast(tf.range(n) + 2, scores.dtype)) /
                              tf.math.log(2.0))
        normalizer = tf.maximum(tf.reduce_sum(ideal_gain * ideal_discount, -1), 1e-12)
        delta = tf.abs((gains[:, :, None] - gains[:, None, :]) *
                       (discount[:, :, None] - discount[:, None, :])) / normalizer[:, None, None]
        weight = tf.stop_gradient(delta * tf.cast(pairs, scores.dtype))
        loss = tf.nn.softplus(-(scores[:, :, None] - scores[:, None, :]))
        return tf.reduce_sum(loss * weight) / tf.maximum(
            tf.reduce_sum(tf.cast(pairs, loss.dtype)), 1)
