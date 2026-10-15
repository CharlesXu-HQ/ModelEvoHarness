"""Small paper-derived graph tensor cores; callers prepare graphs and samples.

These modules do not reproduce graph sampling, training schedules, or serving systems.
"""

import tensorflow as tf


def _attention(logits, mask):
    active = tf.cast(mask, tf.bool)
    work = tf.cast(logits, tf.float32) if logits.dtype in (tf.float16, tf.bfloat16) else logits
    masked = tf.where(active, work, tf.cast(float("-inf"), work.dtype))
    masked = tf.where(tf.reduce_any(active, -1, keepdims=True), masked, tf.zeros_like(masked))
    return tf.cast(tf.where(active, tf.nn.softmax(masked, -1), tf.zeros_like(masked)), logits.dtype)


class GraphSAGE(tf.keras.Model):
    """Mean GraphSAGE layer: call(self[B,D], neighbors[B,K,D], mask[B,K]) -> [B,O].

    Caller supplies sampled neighbors. This is one aggregation layer, not the
    unsupervised random-walk objective from the GraphSAGE paper.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = tf.keras.layers.Dense(output_dim, activation="relu")

    def call(self, self_features, neighbor_features, neighbor_mask):
        active = tf.cast(neighbor_mask, neighbor_features.dtype)
        mean = tf.math.divide_no_nan(tf.reduce_sum(neighbor_features * active[..., None], axis=1),
                                      tf.reduce_sum(active, axis=1, keepdims=True))
        return tf.math.l2_normalize(self.projection(tf.concat((self_features, mean), axis=-1)), axis=-1)


class PinSage(tf.keras.Model):
    """One PinSage convolution; caller provides random-walk visit weights.

    call(self[B,D], neighbors[B,K,D], importance[B,K], mask[B,K]) -> [B,O].
    No walk generation, top-T selection, curriculum negatives, or MapReduce.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.neighbor_projection = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.output_projection = tf.keras.layers.Dense(output_dim, activation="relu")

    def call(self, self_features, neighbor_features, importance, neighbor_mask):
        weights = importance * tf.cast(neighbor_mask, importance.dtype)
        transformed = self.neighbor_projection(neighbor_features)
        pooled = tf.math.divide_no_nan(tf.reduce_sum(transformed * weights[..., None], axis=1),
                                        tf.reduce_sum(weights, axis=1, keepdims=True))
        return tf.math.l2_normalize(self.output_projection(tf.concat((self_features, pooled), axis=-1)), axis=-1)


class SDNE(tf.keras.Model):
    """SDNE autoencoder and first/second-order losses on caller-supplied adjacency.

    call(adjacency_rows[B,N]) -> (embeddings[B,D], reconstruction[B,N]).
    training_loss(adjacency[N,N], edge_index[E,2], edge_weight[E]) -> scalar; edge
    indices address the N rows. Dense adjacency makes this a small-graph core.
    """

    def __init__(self, num_nodes: int, hidden_dim: int, embedding_dim: int,
                 beta: float = 5.0, alpha: float = 1.0, regularization: float = 0.0):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="sigmoid"),
            tf.keras.layers.Dense(embedding_dim, activation="sigmoid")])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="sigmoid"),
            tf.keras.layers.Dense(num_nodes, activation="sigmoid")])
        self.beta, self.alpha, self.regularization = beta, alpha, regularization

    def call(self, adjacency_rows):
        embedding = self.encoder(adjacency_rows)
        return embedding, self.decoder(embedding)

    def training_loss(self, adjacency, edge_index, edge_weight):
        embedding, reconstruction = self(adjacency)
        penalty = tf.where(adjacency != 0, tf.cast(self.beta, adjacency.dtype),
                           tf.ones_like(adjacency))
        second_order = tf.reduce_sum(tf.square((reconstruction - adjacency) * penalty))
        endpoint = tf.gather(embedding, edge_index)
        first_order = tf.reduce_sum(edge_weight * tf.reduce_sum(tf.square(endpoint[:, 0]
                                                                         - endpoint[:, 1]), axis=-1))
        regularizer = tf.add_n([tf.reduce_sum(tf.square(weight)) for weight in self.trainable_variables])
        return second_order + self.alpha * first_order + self.regularization * regularizer


class LINE(tf.keras.Model):
    """First- or second-order LINE with caller-provided weighted edges/negatives.

    training_loss(source[B], target[B], negative_targets[B,K], edge_weight[B]) -> scalar.
    First order shares endpoint embeddings and requires undirected edge semantics;
    second order uses separate source/context embeddings for directed edges.
    Sampling, edge alias tables, and concatenating separately trained orders
    are outside this tensor core.
    """

    def __init__(self, num_nodes: int, embedding_dim: int, order: int = 2):
        super().__init__()
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.order = order
        self.source_embedding = tf.keras.layers.Embedding(num_nodes, embedding_dim)
        self.context_embedding = (self.source_embedding if order == 1
                                  else tf.keras.layers.Embedding(num_nodes, embedding_dim))

    def call(self, source_ids, target_ids):
        return tf.reduce_sum(self.source_embedding(source_ids)
                             * self.context_embedding(target_ids), axis=-1)

    def training_loss(self, source_ids, target_ids, negative_target_ids, edge_weight):
        source = self.source_embedding(source_ids)
        positive = self(source_ids, target_ids)
        negative = tf.reduce_sum(source[:, None, :]
                                 * self.context_embedding(negative_target_ids), axis=-1)
        per_edge = tf.nn.softplus(-positive) + tf.reduce_sum(tf.nn.softplus(negative), axis=-1)
        return tf.math.divide_no_nan(tf.reduce_sum(per_edge * edge_weight),
                                     tf.reduce_sum(edge_weight))


class RippleNet(tf.keras.Model):
    """Multi-hop KG response for candidate items with preassembled ripple triples.

    call(item_ids[B], heads[H,B,M], relations[H,B,M], tails[H,B,M],
    mask[H,B,M]) -> click probabilities[B]. item_update is one of none,
    plus, replace; every hop's response contributes to the user vector.
    This core omits ripple-set construction and the paper's KG auxiliary loss.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 num_hops: int, item_update: str = "none"):
        super().__init__()
        if item_update not in ("none", "plus", "replace"):
            raise ValueError("item_update must be none, plus, or replace")
        self.entity_embedding = tf.keras.layers.Embedding(num_entities, embedding_dim)
        self.relation_embedding = tf.keras.layers.Embedding(num_relations,
                                                             embedding_dim * embedding_dim)
        self.embedding_dim, self.num_hops, self.item_update = embedding_dim, num_hops, item_update

    def call(self, item_ids, heads, relations, tails, mask):
        item = self.entity_embedding(item_ids)
        responses = []
        for hop in range(self.num_hops):
            head = self.entity_embedding(heads[hop])
            tail = self.entity_embedding(tails[hop])
            relation = tf.reshape(self.relation_embedding(relations[hop]),
                                  (tf.shape(head)[0], tf.shape(head)[1],
                                   self.embedding_dim, self.embedding_dim))
            logits = tf.einsum("bi,bmij,bmj->bm", item, relation, head)
            active = tf.cast(mask[hop], tf.bool)
            attention = _attention(logits, active)
            response = tf.reduce_sum(attention[..., None] * tail, axis=1)
            responses.append(response)
            if self.item_update == "plus":
                item = item + response
            elif self.item_update == "replace":
                item = response
        user = tf.add_n(responses)
        return tf.sigmoid(tf.reduce_sum(user * item, axis=-1))


class KGAT(tf.keras.Model):
    """Single relation-aware attention and bi-interaction propagation layer.

    call(head_ids[B], neighbor_ids[B,K], relation_ids[B,K], mask[B,K])
    -> [B,O]. Caller builds the collaborative KG. The TransR KG loss, BPR
    loss, attention-refresh schedule, and deeper layers are not included.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 output_dim: int):
        super().__init__()
        self.entity_embedding = tf.keras.layers.Embedding(num_entities, embedding_dim)
        self.relation_embedding = tf.keras.layers.Embedding(num_relations, embedding_dim)
        self.relation_projection = tf.keras.layers.Embedding(num_relations,
                                                               embedding_dim * embedding_dim)
        self.sum_projection = tf.keras.layers.Dense(output_dim, use_bias=False)
        self.product_projection = tf.keras.layers.Dense(output_dim, use_bias=False)
        self.embedding_dim = embedding_dim

    def call(self, head_ids, neighbor_ids, relation_ids, neighbor_mask):
        head = self.entity_embedding(head_ids)
        tail = self.entity_embedding(neighbor_ids)
        relation = self.relation_embedding(relation_ids)
        projection = tf.reshape(self.relation_projection(relation_ids),
                                (tf.shape(tail)[0], tf.shape(tail)[1],
                                 self.embedding_dim, self.embedding_dim))
        projected_tail = tf.einsum("bkij,bkj->bki", projection, tail)
        projected_head = tf.einsum("bkij,bj->bki", projection, head)
        logits = tf.reduce_sum(projected_tail * tf.tanh(projected_head + relation), axis=-1)
        active = tf.cast(neighbor_mask, tf.bool)
        attention = _attention(logits, active)
        neighborhood = tf.reduce_sum(attention[..., None] * tail, axis=1)
        return (tf.nn.leaky_relu(self.sum_projection(head + neighborhood))
                + tf.nn.leaky_relu(self.product_projection(head * neighborhood)))


class CDL(tf.keras.Model):
    """Deterministic MAP-style CDL content autoencoder coupled to latent MF.

    call(user_ids[B], item_ids[B], corrupted_content[B,C]) ->
    (rating[B], reconstruction[B,C], content_code[B,D]). training_loss additionally
    takes clean_content[B,C], rating[B], confidence[B]. Caller corrupts input
    and supplies confidence weights. This paired minibatch objective repeats
    content/coupling terms for repeated items; it is not the Bayesian sampler.
    """

    def __init__(self, num_users: int, num_items: int, content_dim: int,
                 hidden_dim: int, embedding_dim: int, reconstruction_weight: float = 1.0,
                 coupling_weight: float = 1.0, regularization: float = 0.0):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="sigmoid"),
            tf.keras.layers.Dense(embedding_dim, activation="sigmoid")])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="sigmoid"),
            tf.keras.layers.Dense(content_dim, activation="sigmoid")])
        self.reconstruction_weight = reconstruction_weight
        self.coupling_weight = coupling_weight
        self.regularization = regularization

    def call(self, user_ids, item_ids, corrupted_content):
        code = self.encoder(corrupted_content)
        reconstruction = self.decoder(code)
        rating = tf.reduce_sum(self.user_embedding(user_ids) * self.item_embedding(item_ids), axis=-1)
        return rating, reconstruction, code

    def training_loss(self, user_ids, item_ids, corrupted_content, clean_content, rating, confidence):
        prediction, reconstruction, code = self(user_ids, item_ids, corrupted_content)
        feedback = tf.reduce_mean(confidence * tf.square(prediction - rating))
        content = tf.reduce_mean(tf.reduce_sum(tf.square(reconstruction - clean_content), axis=-1))
        coupling = tf.reduce_mean(tf.reduce_sum(tf.square(self.item_embedding(item_ids) - code), axis=-1))
        regularizer = (tf.reduce_mean(tf.reduce_sum(tf.square(self.user_embedding(user_ids)), axis=-1))
                       + tf.reduce_mean(tf.reduce_sum(tf.square(self.item_embedding(item_ids)), axis=-1)))
        return (feedback + self.reconstruction_weight * content
                + self.coupling_weight * coupling + self.regularization * regularizer)
