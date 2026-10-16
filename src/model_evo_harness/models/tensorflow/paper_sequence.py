"""TensorFlow sequence tensor cores; callers own histories and labels."""

import math

import tensorflow as tf


class _AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim: int, heads: int, name: str):
        super().__init__(name=name)
        self.attention = tf.keras.layers.MultiHeadAttention(heads, dim // heads, dropout=0)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.feedforward = tf.keras.Sequential((tf.keras.layers.Dense(2 * dim, activation="relu"),
                                                tf.keras.layers.Dense(dim)))
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x: tf.Tensor, mask: tf.Tensor, causal: bool = False) -> tf.Tensor:
        length = tf.shape(x)[1]
        safe = tf.where(tf.reduce_any(mask, 1, keepdims=True), mask,
                        tf.one_hot(tf.zeros((tf.shape(mask)[0],), tf.int32), length,
                                   on_value=True, off_value=False, dtype=tf.bool))
        allowed = safe[:, None, :]
        if causal:
            allowed = allowed & tf.linalg.band_part(tf.ones((length, length), tf.bool), -1, 0)[None]
        normalized = self.norm1(x)
        x = x + self.attention(normalized, normalized, attention_mask=allowed)
        return (x + self.feedforward(self.norm2(x))) * tf.cast(mask[..., None], x.dtype)


class TransAct(tf.keras.Model):
    """Candidate-conditioned action encoder; recent K states plus masked max."""

    def __init__(self, dim: int, num_heads: int, recent_k: int):
        super().__init__()
        if recent_k < 1:
            raise ValueError("recent_k must be positive")
        self.recent_k = recent_k
        self.fuse = tf.keras.layers.Dense(dim)
        self.encoder = _AttentionBlock(dim, num_heads, name="attention_block")

    def call(self, history: tf.Tensor, candidate: tf.Tensor,
             mask: tf.Tensor) -> tf.Tensor:
        length = tf.shape(history)[1]
        tf.debugging.assert_positive(length)
        paired = tf.concat((history, tf.broadcast_to(candidate[:, None], tf.shape(history))), -1)
        states = self.encoder(self.fuse(paired) * tf.cast(mask[..., None], history.dtype), mask)
        positions = tf.broadcast_to(tf.range(length)[None], tf.shape(mask))
        order = tf.argsort(tf.where(mask, positions, -1), -1, direction="DESCENDING")
        k = tf.minimum(self.recent_k, length)
        recent_mask = tf.gather(mask, order[:, :k], batch_dims=1)
        recent = tf.gather(states, order[:, :k], batch_dims=1)
        recent = tf.reshape(recent * tf.cast(recent_mask[..., None], states.dtype),
                            (tf.shape(history)[0], -1))
        recent = tf.pad(recent, [[0, 0], [0, (self.recent_k - k) * tf.shape(states)[-1]]])
        maximum = tf.reduce_max(tf.where(mask[..., None], states,
                                         tf.cast(float("-inf"), states.dtype)), 1)
        maximum = tf.where(tf.reduce_any(mask, 1, keepdims=True), maximum, 0)
        return tf.concat((recent, maximum), -1)


class PinnerFormer(tf.keras.Model):
    """Compact causal normalized encoder; caller supplies future targets."""

    def __init__(self, dim: int, num_heads: int, max_len: int):
        super().__init__()
        self.position = tf.keras.layers.Embedding(max_len, dim)
        self.encoder = _AttentionBlock(dim, num_heads, name="attention_block")

    def encode(self, history: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        length = tf.shape(history)[1]
        tf.debugging.assert_positive(length)
        x = history * tf.cast(mask[..., None], history.dtype) + self.position(tf.range(length))[None]
        states = self.encoder(x, mask, causal=True)
        return tf.math.l2_normalize(states, -1) * tf.cast(mask[..., None], states.dtype)

    def call(self, history: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        states = self.encode(history, mask)
        positions = tf.broadcast_to(tf.range(tf.shape(mask)[1])[None], tf.shape(mask))
        last = tf.maximum(tf.reduce_max(tf.where(mask, positions, -1), 1), 0)
        return tf.gather(states, last, batch_dims=1)

    @staticmethod
    def dense_all_action_loss(states: tf.Tensor, positive: tf.Tensor,
                              negatives: tf.Tensor, supervision: tf.Tensor,
                              temperature: float = 1.0) -> tf.Tensor:
        query = tf.math.l2_normalize(states, -1)
        pos = tf.reduce_sum(query * tf.math.l2_normalize(positive, -1), -1,
                            keepdims=True)
        neg = tf.einsum("btd,btnd->btn", query, tf.math.l2_normalize(negatives, -1))
        logits = tf.concat((pos, neg), -1) / temperature
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.zeros(tf.shape(logits)[:2], tf.int32), logits=logits)
        weight = tf.cast(supervision, loss.dtype)
        return tf.reduce_sum(loss * weight) / tf.maximum(tf.reduce_sum(weight), 1)


class Seq2Slate(tf.keras.Model):
    """Teacher-forced pointer logits and greedy unique eligible selections."""

    def __init__(self, dim: int, slate_size: int):
        super().__init__()
        if slate_size < 1:
            raise ValueError("slate_size must be positive")
        self.slate_size = slate_size
        self.key = tf.keras.layers.Dense(dim, use_bias=False)
        self.query = tf.keras.layers.Dense(dim, use_bias=False)
        self.initial = tf.keras.layers.Dense(dim, activation="tanh")
        self.cell = tf.keras.layers.GRUCell(dim)
        self.key.build((None, dim))
        self.query.build((None, dim))
        self.initial.build((None, dim))
        self.cell.build((None, dim))

    def _initial_state(self, candidates: tf.Tensor,
                       eligibility: tf.Tensor) -> tf.Tensor:
        weights = tf.cast(eligibility, candidates.dtype)
        context = tf.reduce_sum(candidates * weights[..., None], 1) / tf.maximum(
            tf.reduce_sum(weights, 1, keepdims=True), 1)
        return self.initial(context)

    def _scores(self, keys: tf.Tensor, state: tf.Tensor,
                available: tf.Tensor) -> tf.Tensor:
        scores = tf.einsum("bnd,bd->bn", keys, self.query(state)) / math.sqrt(keys.shape[-1])
        return tf.where(available, scores, tf.cast(scores.dtype.min, scores.dtype))

    def teacher_forced_logits(self, candidates: tf.Tensor, eligibility: tf.Tensor,
                              targets: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(candidates)[0]
        keys = self.key(candidates)
        state = self._initial_state(candidates, eligibility)
        output = tf.TensorArray(keys.dtype, size=tf.shape(targets)[1])

        def body(step, current, available, logits):
            chosen = targets[:, step]
            tf.debugging.assert_equal(tf.gather(available, chosen, batch_dims=1),
                                      tf.ones((batch,), tf.bool),
                                      message="teacher target is ineligible or repeated")
            logits = logits.write(step, self._scores(keys, current, available))
            selected = tf.gather(candidates, chosen, batch_dims=1)
            _, [current] = self.cell(selected, [current])
            available = available & ~tf.one_hot(chosen, tf.shape(available)[1],
                                                 on_value=True, off_value=False, dtype=tf.bool)
            return step + 1, current, available, logits

        _, _, _, output = tf.while_loop(lambda step, *_: step < tf.shape(targets)[1],
                                        body, (0, state, eligibility, output))
        return tf.cond(tf.shape(targets)[1] > 0,
                       lambda: tf.transpose(output.stack(), (1, 0, 2)),
                       lambda: tf.zeros((batch, 0, tf.shape(candidates)[1]), keys.dtype))

    def call(self, candidates: tf.Tensor, eligibility: tf.Tensor) -> tf.Tensor:
        # One masked dummy slot makes zero-candidate slates safe in traced graphs.
        empty = tf.cast(tf.equal(tf.shape(candidates)[1], 0), tf.int32)
        candidates = tf.pad(candidates, [[0, 0], [0, empty], [0, 0]])
        eligibility = tf.pad(eligibility, [[0, 0], [0, empty]])
        batch = tf.shape(candidates)[0]
        keys = self.key(candidates)
        state = self._initial_state(candidates, eligibility)
        available = eligibility
        result = []
        for _ in range(self.slate_size):
            active = tf.reduce_any(available, 1)
            index = tf.argmax(self._scores(keys, state, available), 1,
                              output_type=tf.int32)
            result.append(tf.where(active, index, -1))
            selected = tf.gather(candidates, index, batch_dims=1)
            _, [new_state] = self.cell(selected, [state])
            state = tf.where(active[:, None], new_state, state)
            available = available & ~tf.one_hot(index, tf.shape(available)[1],
                                                 on_value=True, off_value=False, dtype=tf.bool)
        return tf.stack(result, 1)
