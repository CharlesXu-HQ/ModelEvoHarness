"""Cold-start and bias tensor cores; no data pipeline or training orchestration.

PAL and MetaEmbedding are independent paper-derived implementations. DICE's
masked objectives follow the MIT reference by Zheng et al.; see THIRD_PARTY_NOTICES.md.
"""

# DICE masked objective adapted from FIB LAB, Tsinghua University
# https://github.com/tsinghua-fib-lab/DICE/blob/591b883cd62bf0d88c906d12ed2ab88ff8fccc36/src/model.py
# MIT License
# 
# Copyright (c) 2022 FIB LAB, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf


class PAL(tf.keras.layers.Layer):
    """Position-only examination × relevance; three [B] probabilities.

    The host supplies position-free relevance logits. relevance() is the serving
    operation; the logged-position product trains both factors jointly. The
    assumed factorization does not establish identifiability from arbitrary logs.
    """

    def __init__(self, num_positions: int):
        super().__init__()
        self.examination = tf.keras.layers.Embedding(num_positions, 1, embeddings_initializer="zeros")

    def call(self, relevance_logits, position_ids):
        exposure = tf.sigmoid(tf.squeeze(self.examination(position_ids), -1))
        relevance = self.relevance(relevance_logits)
        return {"relevance": relevance, "examination": exposure,
                "observed": relevance * exposure}

    def relevance(self, relevance_logits):
        return tf.sigmoid(relevance_logits)

    def objective(self, relevance_logits, position_ids, labels):
        """Mean joint BCE, stable even when both sigmoid outputs round to one."""
        a, b = relevance_logits, tf.squeeze(self.examination(position_ids), -1)
        joint_logit = a + b - tf.reduce_logsumexp(tf.stack((tf.zeros_like(a), a, b)), 0)
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=joint_logit))


class MetaEmbedding(tf.keras.layers.Layer):
    """Frozen attributes [B,A] -> generated ID vectors [B,D]; not a meta-trainer.

    Host owns per-entity support/query batches and nested GradientTapes through
    adapt(). Plain supervised training does not reproduce the meta objective.
    """

    def __init__(self, attribute_dim: int, embedding_dim: int):
        super().__init__()
        self.generator = tf.keras.layers.Dense(embedding_dim, activation="tanh", use_bias=False)
        self.generator.build((None, attribute_dim))

    def call(self, attributes):
        return self.generator(tf.stop_gradient(attributes))

    def adapt(self, embedding, support_gradient, step_size: float):
        """Functional inner update; calculate support gradient inside the outer tape."""
        return embedding - step_size * support_gradient

    def meta_loss(self, cold_loss, warm_loss, alpha: float = 0.1):
        return alpha * cold_loss + (1 - alpha) * warm_loss


class DICE(tf.keras.layers.Layer):
    """Interest/conformity factors; ID pairs [B] -> three [B] scores.

    The objective mask is true when the negative item is more popular. Host owns
    that sampling contract, curriculum and regularization. The optional negative
    L2 discrepancy is unbounded without regularization and defaults to weight 0.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_interest = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.user_conformity = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_interest = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.item_conformity = tf.keras.layers.Embedding(num_items, embedding_dim)

    def call(self, user_ids, item_ids):
        interest = tf.reduce_sum(self.user_interest(user_ids) * self.item_interest(item_ids), -1)
        conformity = tf.reduce_sum(self.user_conformity(user_ids) * self.item_conformity(item_ids), -1)
        return {"interest": interest, "conformity": conformity, "total": interest + conformity}

    def objective(self, user_ids, positive_ids, negative_ids, negative_more_popular,
                  interest_weight: float = 0.1, conformity_weight: float = 0.1,
                  discrepancy_weight: float = 0.0):
        p, n = self(user_ids, positive_ids), self(user_ids, negative_ids)
        mask = tf.cast(negative_more_popular, p["total"].dtype)
        interest = tf.reduce_mean(mask * tf.nn.softplus(n["interest"] - p["interest"]))
        conformity = tf.reduce_mean(mask * tf.nn.softplus(p["conformity"] - n["conformity"])
                                   + (1 - mask) * tf.nn.softplus(n["conformity"] - p["conformity"]))
        users = tf.unique(user_ids).y
        items = tf.unique(tf.concat((positive_ids, negative_ids), 0)).y
        discrepancy = tf.reduce_mean(tf.square(self.user_interest(users) - self.user_conformity(users)))
        discrepancy += tf.reduce_mean(tf.square(self.item_interest(items) - self.item_conformity(items)))
        return (tf.reduce_mean(tf.nn.softplus(n["total"] - p["total"])) + interest_weight * interest
                + conformity_weight * conformity - discrepancy_weight * discrepancy)
