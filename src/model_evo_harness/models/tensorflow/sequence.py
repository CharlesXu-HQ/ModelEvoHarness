"""Direct TensorFlow sequence and interest-model cores.

Inputs are pre-decision embeddings; the host owns IDs, labels, loss, and evaluation.
"""

import math

import tensorflow as tf


def _mean(values: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    weights = tf.cast(mask[..., None], values.dtype)
    return tf.reduce_sum(values * weights, axis=1) / tf.maximum(
        tf.reduce_sum(weights, axis=1), 1)


def _last(values: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    positions = tf.range(tf.shape(values)[1])[None]
    index = tf.maximum(tf.reduce_max(tf.where(mask, positions, -1), axis=1), 0)
    selected = tf.gather(values, index, axis=1, batch_dims=1)
    return selected * tf.cast(tf.reduce_any(mask, axis=1, keepdims=True), values.dtype)


def _require_prefix(mask: tf.Tensor) -> tf.Tensor:
    """Recurrent encoders accept left-aligned valid events followed by padding."""
    invalid = mask[:, 1:] & ~mask[:, :-1]
    assertion = tf.debugging.assert_equal(invalid, tf.zeros_like(invalid),
                                          message="recurrent history mask must be a prefix")
    with tf.control_dependencies([assertion] if assertion is not None else []):
        return tf.identity(mask)


def _attention(scores: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    weights = tf.nn.softmax(tf.where(mask, scores, tf.cast(-1e9, scores.dtype)), axis=-1)
    weights *= tf.cast(mask, scores.dtype)
    return weights / tf.maximum(tf.reduce_sum(weights, axis=-1, keepdims=True), 1e-9)


class MIND(tf.keras.Model):
    """Dynamic routing from history embeddings [B,T,D] to K interest vectors."""

    def __init__(self, dim: int, num_interests: int, routing_iters: int = 3):
        super().__init__()
        self.prototypes = self.add_weight(
            name="prototypes", shape=(num_interests, dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=1 / math.sqrt(dim)))
        self.routing_iters = routing_iters

    def call(self, history: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        logits = tf.einsum("btd,kd->btk", history, self.prototypes) / tf.sqrt(
            tf.cast(tf.shape(history)[-1], history.dtype))
        for _ in range(self.routing_iters):
            assignment = tf.nn.softmax(logits, axis=-1) * tf.cast(mask[..., None], history.dtype)
            interests = tf.einsum("btk,btd->bkd", assignment, history)
            count = tf.reduce_sum(assignment, axis=1)[..., None]
            interests = tf.math.l2_normalize(
                interests / tf.maximum(count, 1) + self.prototypes[None], axis=-1)
            logits += tf.einsum("btd,bkd->btk", history, interests)
        return interests

    @staticmethod
    def score(interests: tf.Tensor, items: tf.Tensor) -> tf.Tensor:
        return tf.reduce_max(tf.einsum("bkd,id->bki", interests, items), axis=1)

    @staticmethod
    def label_aware_weights(interests: tf.Tensor, target: tf.Tensor,
                            temperature: float = 1.0) -> tf.Tensor:
        """Training-only target-conditioned weights over K interests [B,K,D]."""
        return tf.nn.softmax(tf.einsum("bkd,bd->bk", interests, target) / temperature,
                             axis=1)

    def training_readout(self, history: tf.Tensor, mask: tf.Tensor,
                         target: tf.Tensor, temperature: float = 1.0) -> tf.Tensor:
        """Return a target-aware user vector; inference can retain all K interests."""
        interests = self(history, mask)
        weights = self.label_aware_weights(interests, target, temperature)
        return tf.einsum("bk,bkd->bd", weights, interests)


class SASRec(tf.keras.Model):
    """Causal self-attention for ordered history embeddings [B,T,D]."""

    def __init__(self, dim: int, num_heads: int, max_len: int):
        super().__init__()
        self.position = tf.keras.layers.Embedding(max_len, dim)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                            key_dim=dim // num_heads)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.feedforward = tf.keras.Sequential([
            tf.keras.layers.Dense(2 * dim, activation="relu"),
            tf.keras.layers.Dense(dim),
        ])
        self.norm2 = tf.keras.layers.LayerNormalization()

    def encode(self, history: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        length = tf.shape(history)[1]
        x = history + self.position(tf.range(length))[None]
        causal = tf.linalg.band_part(tf.ones((length, length), dtype=tf.bool), -1, 0)
        safe = tf.where(tf.reduce_any(mask, axis=1, keepdims=True), mask,
                        tf.one_hot(tf.zeros((tf.shape(mask)[0],), dtype=tf.int32),
                                   length, on_value=True, off_value=False, dtype=tf.bool))
        allowed = causal[None] & safe[:, None, :]
        x = self.norm1(x + self.attention(x, x, attention_mask=allowed))
        return self.norm2(x + self.feedforward(x)) * tf.cast(mask[..., None], x.dtype)

    def call(self, history: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        return _last(self.encode(history, mask), mask)


class SDM(tf.keras.Model):
    """Short LSTM/self-attention and long user-attention with gated fusion.

    Histories are [B,T,D] with contiguous-prefix masks. Omitted long history
    reuses short history; omitted user context is the masked long-history mean.
    """

    def __init__(self, dim: int, short_window: int):
        super().__init__()
        self.short_window = short_window
        heads = 2 if dim % 2 == 0 else 1
        self.short_lstm = tf.keras.layers.LSTM(dim, return_sequences=True)
        self.short_attention = tf.keras.layers.MultiHeadAttention(num_heads=heads,
                                                                  key_dim=dim // heads)
        self.short_query = tf.keras.layers.Dense(dim, use_bias=False)
        self.long_query = tf.keras.layers.Dense(dim, use_bias=False)
        self.gate = tf.keras.layers.Dense(dim, activation="sigmoid")

    def call(self, short_history: tf.Tensor, short_mask: tf.Tensor,
             long_history: tf.Tensor | None = None, long_mask: tf.Tensor | None = None,
             user_context: tf.Tensor | None = None) -> tf.Tensor:
        if long_history is None:
            long_history, long_mask = short_history, short_mask
        if long_mask is None:
            raise ValueError("long_mask is required with long_history")
        short_mask = _require_prefix(short_mask)
        long_mask = _require_prefix(long_mask)
        if user_context is None:
            user_context = _mean(long_history, long_mask)

        window = tf.minimum(self.short_window, tf.shape(short_history)[1])
        lengths = tf.reduce_sum(tf.cast(short_mask, tf.int32), axis=1)
        offsets = tf.range(window)[None]
        indices = tf.maximum(lengths - window, 0)[:, None] + offsets
        recent = tf.gather(short_history, indices, axis=1, batch_dims=1)
        recent_mask = offsets < tf.minimum(lengths, window)[:, None]
        recent *= tf.cast(recent_mask[..., None], recent.dtype)
        short_states = self.short_lstm(recent, mask=recent_mask)
        safe = tf.where(tf.reduce_any(recent_mask, axis=1, keepdims=True), recent_mask,
                        tf.one_hot(tf.zeros((tf.shape(recent_mask)[0],), dtype=tf.int32),
                                   window, on_value=True, off_value=False, dtype=tf.bool))
        allowed = tf.broadcast_to(safe[:, None, :],
                                  (tf.shape(safe)[0], window, window))
        attended = self.short_attention(short_states, short_states,
                                         attention_mask=allowed)
        dim = tf.cast(tf.shape(short_history)[-1], short_history.dtype)
        short_scores = tf.einsum("btd,bd->bt", attended,
                                 self.short_query(user_context)) / tf.sqrt(dim)
        short = tf.einsum("bt,btd->bd", _attention(short_scores, recent_mask), attended)
        long_scores = tf.einsum("btd,bd->bt", long_history,
                                self.long_query(user_context)) / tf.sqrt(dim)
        long = tf.einsum("bt,btd->bd", _attention(long_scores, long_mask), long_history)
        gate = self.gate(tf.concat([short, long, user_context], axis=-1))
        output = gate * short + (1 - gate) * long
        has_history = tf.reduce_any(short_mask, axis=1) | tf.reduce_any(long_mask, axis=1)
        return output * tf.cast(has_history[:, None], output.dtype)


class NARM(tf.keras.Model):
    """GRU session encoder with final-state-conditioned local attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.gru = tf.keras.layers.GRU(dim, return_sequences=True)
        self.query = tf.keras.layers.Dense(dim, use_bias=False)
        self.key = tf.keras.layers.Dense(dim, use_bias=False)
        self.weight = tf.keras.layers.Dense(1, use_bias=False)
        self.fuse = tf.keras.layers.Dense(dim)

    def call(self, session: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        mask = _require_prefix(mask)
        states = self.gru(session * tf.cast(mask[..., None], session.dtype), mask=mask)
        global_state = _last(states, mask)
        scores = tf.squeeze(self.weight(tf.sigmoid(self.key(states) +
                                                   self.query(global_state)[:, None])), axis=-1)
        local = tf.einsum("bt,btd->bd", _attention(scores, mask), states)
        output = self.fuse(tf.concat([global_state, local], axis=-1))
        return output * tf.cast(tf.reduce_any(mask, axis=1, keepdims=True), output.dtype)


class HSTU(tf.keras.Model):
    """Causal gated pointwise sequence unit with relative-time attention bias."""

    def __init__(self, dim: int, max_time_gap: int = 128):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization()
        self.qkvu = tf.keras.layers.Dense(4 * dim)
        self.time_bias = tf.keras.layers.Embedding(max_time_gap + 1, 1)
        self.out = tf.keras.layers.Dense(dim)
        self.max_time_gap = max_time_gap

    def call(self, history: tf.Tensor, timestamps: tf.Tensor,
             mask: tf.Tensor) -> tf.Tensor:
        q, k, v, u = tf.split(self.qkvu(self.norm(history)), 4, axis=-1)
        logits = tf.matmul(q, k, transpose_b=True) / tf.sqrt(
            tf.cast(tf.shape(history)[-1], history.dtype))
        gap = tf.clip_by_value(timestamps[:, :, None] - timestamps[:, None, :],
                               0, self.max_time_gap)
        logits += tf.squeeze(self.time_bias(tf.cast(gap, tf.int32)), axis=-1)
        length = tf.shape(history)[1]
        causal = tf.linalg.band_part(tf.ones((length, length), dtype=tf.bool), -1, 0)
        valid = causal[None] & mask[:, :, None] & mask[:, None, :]
        weights = tf.nn.silu(logits) * tf.cast(valid, logits.dtype) / tf.sqrt(
            tf.cast(length, logits.dtype))
        output = history + self.out(tf.matmul(weights, v) * tf.sigmoid(u))
        return output * tf.cast(mask[..., None], history.dtype)


class _AUGRUCell(tf.keras.layers.Layer):
    def __init__(self, dim: int):
        super().__init__()
        self.gates = tf.keras.layers.Dense(2 * dim)
        self.candidate = tf.keras.layers.Dense(dim)

    def call(self, x: tf.Tensor, previous: tf.Tensor,
             attention: tf.Tensor) -> tf.Tensor:
        reset, update = tf.split(tf.sigmoid(self.gates(tf.concat([x, previous], -1))),
                                 2, axis=-1)
        proposal = tf.tanh(self.candidate(tf.concat([x, reset * previous], -1)))
        return previous + update * attention[:, None] * (proposal - previous)


class DIEN(tf.keras.Model):
    """GRU interest extraction and target-aware AUGRU interest evolution."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.extractor = tf.keras.layers.GRU(dim, return_sequences=True)
        self.evolution = _AUGRUCell(dim)
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, target: tf.Tensor, history: tf.Tensor,
             mask: tf.Tensor) -> tf.Tensor:
        interests = self.extractor(history, mask=mask)
        attention = _attention(tf.einsum("btd,bd->bt", interests, target), mask)
        def step(index: tf.Tensor, state: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            evolved = self.evolution(interests[:, index], state, attention[:, index])
            state = tf.where(mask[:, index, None], evolved, state)
            return index + 1, state

        _, state = tf.while_loop(lambda index, _: index < tf.shape(history)[1],
                                 step, (tf.constant(0), tf.zeros_like(target)))
        return tf.squeeze(self.head(tf.concat([target, state], axis=-1)), axis=-1)


class DSIN(tf.keras.Model):
    """Session self-attention, cross-session BiLSTM, and target activation."""

    def __init__(self, dim: int, hidden_dim: int, heads: int):
        super().__init__()
        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=heads,
                                                                 key_dim=dim // heads)
        self.evolution = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(dim, return_sequences=True))
        self.evolution_query = tf.keras.layers.Dense(2 * dim, use_bias=False)
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, target: tf.Tensor, sessions: tf.Tensor,
             mask: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(sessions)
        batch, count, length = shape[0], shape[1], shape[2]
        flat = tf.reshape(sessions, (batch * count, length, shape[3]))
        flat_mask = tf.reshape(mask, (batch * count, length))
        safe = tf.where(tf.reduce_any(flat_mask, axis=1, keepdims=True), flat_mask,
                        tf.one_hot(tf.zeros((batch * count,), dtype=tf.int32), length,
                                   on_value=True, off_value=False, dtype=tf.bool))
        allowed = tf.broadcast_to(safe[:, None, :], (batch * count, length, length))
        attended = self.self_attention(flat, flat, attention_mask=allowed)
        pooled = tf.reshape(_mean(attended, flat_mask), (batch, count, shape[3]))
        session_mask = tf.reduce_any(mask, axis=-1)
        evolved = self.evolution(pooled, mask=session_mask)
        scores = tf.einsum("bsd,bd->bs", pooled, target)
        evolved_scores = tf.einsum("bsd,bd->bs", evolved,
                                   self.evolution_query(target))
        local = tf.einsum("bs,bsd->bd", _attention(scores, session_mask), pooled)
        temporal = tf.einsum("bs,bsd->bd", _attention(evolved_scores, session_mask),
                             evolved)
        return tf.squeeze(self.head(tf.concat([target, local, temporal], axis=-1)), axis=-1)
