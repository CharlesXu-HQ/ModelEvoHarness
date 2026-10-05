"""Independent TensorFlow references for collaborative filtering and retrieval.

Inputs are already encoded tensors; host code owns positive-event filtering,
temporal splitting, negative sampling, and catalog indexing.
"""

import tensorflow as tf


class FunkSVD(tf.keras.Model):
    """Pair score from user/item ID embeddings; call(user_ids, item_ids) -> [B]."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)

    def call(self, user_ids: tf.Tensor, item_ids: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(self.user_embedding(user_ids) * self.item_embedding(item_ids),
                             axis=-1)


class BiasSVD(FunkSVD):
    """FunkSVD plus user, item, and global offsets; pair logits [B]."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__(num_users, num_items, embedding_dim)
        self.user_bias = tf.keras.layers.Embedding(num_users, 1,
                                                   embeddings_initializer="zeros")
        self.item_bias = tf.keras.layers.Embedding(num_items, 1,
                                                   embeddings_initializer="zeros")
        self.global_bias = self.add_weight(name="global_bias", shape=(),
                                           initializer="zeros")

    def call(self, user_ids: tf.Tensor, item_ids: tf.Tensor) -> tf.Tensor:
        return (super().call(user_ids, item_ids)
                + tf.squeeze(self.user_bias(user_ids), axis=-1)
                + tf.squeeze(self.item_bias(item_ids), axis=-1) + self.global_bias)


class ItemCF:
    """Positive-only cosine item neighbors; fit([U,I]), scores([B,I]) -> [B,I]."""

    def fit(self, interactions: tf.Tensor) -> "ItemCF":
        positive = tf.cast(interactions > 0, tf.float32)
        counts = tf.reduce_sum(positive, axis=0)
        scale = tf.maximum(tf.sqrt(counts[:, None] * counts[None, :]), 1)
        similarity = tf.matmul(positive, positive, transpose_a=True) / scale
        self.similarity = tf.linalg.set_diag(similarity, tf.zeros_like(counts))
        return self

    def scores(self, user_histories: tf.Tensor) -> tf.Tensor:
        return tf.matmul(tf.cast(user_histories, tf.float32), self.similarity)


class UserCF:
    """Positive-only cosine user neighbors; fit([U,I]), scores(user_ids) -> [B,I]."""

    def fit(self, interactions: tf.Tensor) -> "UserCF":
        self.interactions = tf.cast(interactions > 0, tf.float32)
        counts = tf.reduce_sum(self.interactions, axis=1)
        scale = tf.maximum(tf.sqrt(counts[:, None] * counts[None, :]), 1)
        similarity = tf.matmul(self.interactions, self.interactions, transpose_b=True) / scale
        self.similarity = tf.linalg.set_diag(similarity, tf.zeros_like(counts))
        return self

    def scores(self, user_ids: tf.Tensor) -> tf.Tensor:
        return tf.matmul(tf.gather(self.similarity, user_ids), self.interactions)


class Swing:
    """Common-user-pair item similarity; fit([U,I]), scores([B,I]) -> [B,I].

    Dense eager reference for small item catalogs. Production use needs
    candidate-pair pruning and a sparse implementation.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, interactions: tf.Tensor) -> "Swing":
        positive = tf.cast(interactions > 0, tf.float32)
        activity = tf.reduce_sum(positive, axis=1)
        shared = tf.matmul(positive, positive, transpose_b=True)
        pair_weight = tf.math.rsqrt(tf.maximum(activity[:, None] * activity[None, :], 1))
        pair_weight = pair_weight / (self.alpha + shared)
        pair_weight = tf.linalg.set_diag(pair_weight, tf.zeros_like(activity))
        item_count = positive.shape[1]
        rows = []
        for i in range(item_count):
            row = []
            for j in range(item_count):
                common = positive[:, i] * positive[:, j]
                score = 0.5 * tf.tensordot(common, tf.linalg.matvec(pair_weight, common),
                                           axes=1)
                row.append(tf.zeros_like(score) if i == j else score)
            rows.append(tf.stack(row))
        self.similarity = tf.stack(rows)
        return self

    def scores(self, user_histories: tf.Tensor) -> tf.Tensor:
        return tf.matmul(tf.cast(user_histories, tf.float32), self.similarity)


class EGES(tf.keras.Model):
    """Item-specific mixture of ID/side embeddings scored against a context ID.

    call(item_ids[B], side_ids[B,S], context_ids[B]) -> pair logits [B].
    """

    def __init__(self, num_items: int, side_cardinalities: list[int],
                 embedding_dim: int):
        super().__init__()
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.side_embeddings = [tf.keras.layers.Embedding(n, embedding_dim)
                                for n in side_cardinalities]
        self.attention = tf.keras.layers.Embedding(num_items,
                                                    len(side_cardinalities) + 1)
        self.context_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)

    def attention_weights(self, item_ids: tf.Tensor) -> tf.Tensor:
        return tf.nn.softmax(self.attention(item_ids), axis=-1)

    def encode_items(self, item_ids: tf.Tensor, side_ids: tf.Tensor) -> tf.Tensor:
        sources = tf.stack([self.item_embedding(item_ids)] + [
            layer(side_ids[:, i]) for i, layer in enumerate(self.side_embeddings)], axis=1)
        return tf.reduce_sum(sources * self.attention_weights(item_ids)[:, :, None], axis=1)

    def call(self, item_ids: tf.Tensor, side_ids: tf.Tensor,
             context_ids: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(self.encode_items(item_ids, side_ids)
                             * self.context_embedding(context_ids), axis=-1)


class Item2Vec(tf.keras.Model):
    """Skip-gram with negative sampling on ordered item-context pairs.

    negative_sampling_loss(center_ids[B], positive_ids[B], negative_ids[B,K]) -> scalar.
    user_vector(history_ids[B,T], mask[B,T]) -> mean input vector [B,D].
    """

    def __init__(self, num_items: int, embedding_dim: int):
        super().__init__()
        self.input_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.output_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)

    def call(self, center_ids: tf.Tensor, context_ids: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(self.input_embedding(center_ids)
                             * self.output_embedding(context_ids), axis=-1)

    def negative_sampling_loss(self, center_ids: tf.Tensor,
                               positive_ids: tf.Tensor,
                               negative_ids: tf.Tensor) -> tf.Tensor:
        center = self.input_embedding(center_ids)
        positive = self(center_ids, positive_ids)
        negatives = tf.reduce_sum(center[:, None, :]
                                  * self.output_embedding(negative_ids), axis=-1)
        return tf.reduce_mean(tf.nn.softplus(-positive)
                              + tf.reduce_sum(tf.nn.softplus(negatives), axis=1))

    def user_vector(self, history_ids: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        vectors = self.input_embedding(history_ids)
        active = tf.cast(mask, vectors.dtype)
        return tf.math.divide_no_nan(tf.reduce_sum(vectors * active[:, :, None], axis=1),
                                     tf.reduce_sum(active, axis=1, keepdims=True))


class DSSM(tf.keras.Model):
    """Independent MLP towers with cosine pair score; call([B,Du],[B,Di]) -> [B]."""

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

    def encode_user(self, features: tf.Tensor) -> tf.Tensor:
        return tf.math.l2_normalize(self.user_tower(features), axis=-1)

    def encode_item(self, features: tf.Tensor) -> tf.Tensor:
        return tf.math.l2_normalize(self.item_tower(features), axis=-1)

    def call(self, user_features: tf.Tensor,
             item_features: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(self.encode_user(user_features)
                             * self.encode_item(item_features), axis=-1)


class FMRecall(tf.keras.Model):
    """Decomposed FM matching towers for one categorical ID per field.

    encode_user([B,U]), encode_item([B,I]) -> [B,D+1]; paired call -> [B].
    The item tower carries item linear and item-item interaction terms.
    """

    def __init__(self, user_cardinalities: list[int], item_cardinalities: list[int],
                 embedding_dim: int):
        super().__init__()
        self.user_embeddings = [tf.keras.layers.Embedding(n, embedding_dim)
                                for n in user_cardinalities]
        self.item_embeddings = [tf.keras.layers.Embedding(n, embedding_dim)
                                for n in item_cardinalities]
        self.item_linear = [tf.keras.layers.Embedding(n, 1)
                            for n in item_cardinalities]

    def encode_user(self, fields: tf.Tensor) -> tf.Tensor:
        vectors = tf.stack([layer(fields[:, i])
                            for i, layer in enumerate(self.user_embeddings)], axis=1)
        return tf.concat([tf.ones((tf.shape(fields)[0], 1), dtype=vectors.dtype),
                          tf.reduce_sum(vectors, axis=1)], axis=1)

    def encode_item(self, fields: tf.Tensor) -> tf.Tensor:
        vectors = tf.stack([layer(fields[:, i])
                            for i, layer in enumerate(self.item_embeddings)], axis=1)
        summed = tf.reduce_sum(vectors, axis=1)
        pair_terms = 0.5 * tf.reduce_sum(tf.square(summed)
                                          - tf.reduce_sum(tf.square(vectors), axis=1),
                                          axis=1)
        linear = tf.add_n([tf.squeeze(layer(fields[:, i]), axis=-1)
                           for i, layer in enumerate(self.item_linear)])
        return tf.concat([(linear + pair_terms)[:, None], summed], axis=1)

    def call(self, user_fields: tf.Tensor, item_fields: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(self.encode_user(user_fields)
                             * self.encode_item(item_fields), axis=-1)


class YouTubeDNN(tf.keras.Model):
    """User DNN and item-ID table for sampled-candidate retrieval.

    call(user_features[B,Du], candidate_ids[K], history_ids[B,T] or None,
         history_mask[B,T] or None) -> scores[B,K]. Host chooses the sampler
    and cross-entropy target; this is not full sampled softmax.
    """

    def __init__(self, user_dim: int, num_items: int, embedding_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.history_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.user_tower = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ])
        self.embedding_dim = embedding_dim

    def encode_user(self, user_features: tf.Tensor,
                    history_ids: tf.Tensor | None = None,
                    history_mask: tf.Tensor | None = None) -> tf.Tensor:
        if history_ids is None:
            pooled = tf.zeros((tf.shape(user_features)[0], self.embedding_dim),
                              dtype=user_features.dtype)
        else:
            vectors = self.history_embedding(history_ids)
            active = (tf.ones_like(history_ids, dtype=vectors.dtype)
                      if history_mask is None else tf.cast(history_mask, vectors.dtype))
            pooled = tf.math.divide_no_nan(
                tf.reduce_sum(vectors * active[:, :, None], axis=1),
                tf.reduce_sum(active, axis=1, keepdims=True))
        return tf.math.l2_normalize(self.user_tower(tf.concat([user_features, pooled],
                                                           axis=1)), axis=-1)

    def encode_item(self, item_ids: tf.Tensor) -> tf.Tensor:
        return self.item_embedding(item_ids)

    def call(self, user_features: tf.Tensor, candidate_ids: tf.Tensor,
             history_ids: tf.Tensor | None = None,
             history_mask: tf.Tensor | None = None) -> tf.Tensor:
        return tf.matmul(self.encode_user(user_features, history_ids, history_mask),
                         self.encode_item(candidate_ids), transpose_b=True)


class YouTubeSBC(YouTubeDNN):
    """YouTubeDNN with log sampling-probability correction on the same candidates.

    call(user_features, candidate_ids[K], sampling_probs[K], history_ids,
         history_mask) -> corrected scores[B,K]. These are training-sampler
    probabilities, not exposure or treatment-assignment propensities.
    """

    def call(self, user_features: tf.Tensor, candidate_ids: tf.Tensor,
             sampling_probs: tf.Tensor,
             history_ids: tf.Tensor | None = None,
             history_mask: tf.Tensor | None = None) -> tf.Tensor:
        tf.debugging.assert_positive(sampling_probs)
        return (super().call(user_features, candidate_ids, history_ids, history_mask)
                - tf.math.log(sampling_probs)[None, :])
