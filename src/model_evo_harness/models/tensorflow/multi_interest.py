"""Independent native ComiRec interest extractors (https://arxiv.org/abs/2005.09347).

SA follows paper attention with the author's score-only positional option. DR
uses position/interest projections and stopped non-final votes described by
THUDM/ComiRec, src/model.py at 7e605f1. No third-party model code is included.
The host owns embeddings, training objectives, candidate readout and retrieval.
Inputs must be floating history [B,T,D] and bool valid [B,T], True=valid. Valid
values must be finite; masked NaN/Inf padding is sanitized before projection.
"""
import math
from numbers import Integral

import tensorflow as tf


def _positive(**sizes):
    for name, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _history(history, valid, dim, max_length):
    if valid.dtype != tf.bool:
        raise TypeError("valid must be bool with True=valid")
    if not history.dtype.is_floating:
        raise TypeError("history must have floating dtype")
    tf.debugging.assert_rank(history, 3, message="history must have rank 3")
    tf.debugging.assert_rank(valid, 2, message="valid must have rank 2")
    tf.debugging.assert_equal(tf.shape(history)[-1], dim, message="history width mismatch")
    tf.debugging.assert_equal(tf.shape(history)[:2], tf.shape(valid), message="history/valid shape mismatch")
    if max_length is not None:
        tf.debugging.assert_less_equal(tf.shape(history)[1], max_length, message="history length exceeds max_length")
    return tf.where(valid[..., None], history, tf.zeros_like(history))


def _squash(values):
    work = tf.cast(values, tf.float32) if values.dtype in (tf.float16, tf.bfloat16) else values
    squared_norm = tf.reduce_sum(tf.square(work), -1, keepdims=True)
    return tf.cast(squared_norm / (1 + squared_norm) * work / tf.sqrt(squared_norm + 1e-9), values.dtype)


class ComiRecSA(tf.keras.layers.Layer):
    """Candidate-independent self-attentive interests.

    Bias-free hidden -> tanh -> score layers produce K logits per event;
    hidden_dim defaults to dim. If max_length is set, trainable position [T,D]
    is added only when computing scores. Values remain the original history.
    Returns interests [B,K,D], interest_mask [B,K], assignments [B,K,T]. Each
    supported interest's assignments sum to one over T. interest_mask reports
    history presence for every K slot, not distinct-cluster usage. Empty or
    all-masked histories return exact zeros and False masks. Position indices
    are physical input slots; the host owns ordering and time-safe truncation.
    """

    def __init__(self, dim: int, num_interests: int, hidden_dim: int | None = None,
                 max_length: int | None = None):
        super().__init__()
        hidden_dim = dim if hidden_dim is None else hidden_dim
        _positive(dim=dim, num_interests=num_interests, hidden_dim=hidden_dim)
        if max_length is not None:
            _positive(max_length=max_length)
        self.dim, self.num_interests, self.max_length = dim, num_interests, max_length
        self.hidden = tf.keras.layers.Dense(hidden_dim, use_bias=False)
        self.score = tf.keras.layers.Dense(num_interests, use_bias=False)
        self.hidden.build((None, dim))
        self.score.build((None, hidden_dim))
        self.position = None if max_length is None else self.add_weight(
            name="position", shape=(max_length, dim), initializer=tf.keras.initializers.RandomNormal(stddev=.02))

    def call(self, history, valid):
        values = _history(history, valid, self.dim, self.max_length)
        score_input = values if self.position is None else values + self.position[:tf.shape(values)[1]]
        scores = tf.transpose(self.score(tf.tanh(self.hidden(score_input))), [0, 2, 1])
        if scores.dtype in (tf.float16, tf.bfloat16):
            scores = tf.cast(scores, tf.float32)
        masked = tf.where(valid[:, None], scores, tf.cast(float("-inf"), scores.dtype))
        present = tf.reduce_any(valid, -1, keepdims=True)
        # A sentinel keeps softmax finite for T=0 or all-masked histories.
        sentinel = tf.where(present[:, None], tf.cast(float("-inf"), scores.dtype), tf.cast(0., scores.dtype))
        sentinel = tf.broadcast_to(sentinel, [tf.shape(values)[0], self.num_interests, 1])
        assignments = tf.cast(tf.nn.softmax(tf.concat([masked, sentinel], -1), -1)[..., :-1], values.dtype)
        return dict(interests=tf.matmul(assignments, values),
                    interest_mask=tf.broadcast_to(present, [tf.shape(values)[0], self.num_interests]),
                    assignments=assignments)


class ComiRecDR(tf.keras.layers.Layer):
    """Position-specific capsule routing with deterministic zero initial logits.

    projection[t,k,input,output] is trainable [max_length,K,D,D]. Votes are
    routed by softmax over K for each event, summed over T, then squashed as
    ||s||^2/(1+||s||^2) * s/sqrt(||s||^2+1e-9). Unlike SA, assignments sum to
    one over K at each valid event; invalid events have zero coefficients.
    With stop_gradient_routing=True, non-final votes are detached for both
    capsule sums and agreement updates. The final weighted sum uses live
    votes, so history/projection gradients still flow. False differentiates
    through all routing iterations. Zero logits replace author-code random
    initialization; no sampled logits or length-dependent interest count.

    Returns interests [B,K,D], interest_mask [B,K], assignments [B,K,T]. The
    mask reports any valid history for every K slot, not cluster occupancy.
    Empty or all-masked histories return exact-zero vectors and coefficients.
    """

    def __init__(self, dim: int, num_interests: int, max_length: int,
                 routing_iterations: int = 3, stop_gradient_routing: bool = True):
        super().__init__()
        _positive(dim=dim, num_interests=num_interests, max_length=max_length,
                  routing_iterations=routing_iterations)
        if not isinstance(stop_gradient_routing, bool):
            raise TypeError("stop_gradient_routing must be bool")
        self.dim, self.num_interests, self.max_length = dim, num_interests, max_length
        self.routing_iterations, self.stop_gradient_routing = routing_iterations, stop_gradient_routing
        self.projection = self.add_weight(name="projection", shape=(max_length, num_interests, dim, dim),
                                         initializer=tf.keras.initializers.RandomNormal(stddev=1/math.sqrt(dim)))

    def call(self, history, valid):
        history = _history(history, valid, self.dim, self.max_length)
        votes = tf.einsum("btd,tkde->bkte", history, self.projection[:tf.shape(history)[1]])
        # Accumulate routing in float32 for low-precision callers.
        work = tf.cast(votes, tf.float32) if votes.dtype in (tf.float16, tf.bfloat16) else votes
        routing_votes = tf.stop_gradient(work) if self.stop_gradient_routing else work
        logits = tf.zeros(tf.shape(work)[:-1], work.dtype)
        for iteration in range(self.routing_iterations):
            assignments = tf.nn.softmax(logits, axis=1) * tf.cast(valid[:, None], work.dtype)
            final = iteration == self.routing_iterations - 1
            active_votes = work if final else routing_votes
            interests = _squash(tf.reduce_sum(assignments[..., None] * active_votes, 2))
            if not final:
                logits = logits + tf.reduce_sum(routing_votes * interests[:, :, None], -1)
        return dict(interests=tf.cast(interests, votes.dtype),
                    interest_mask=tf.broadcast_to(tf.reduce_any(valid, -1, keepdims=True),
                                                  [tf.shape(history)[0], self.num_interests]),
                    assignments=tf.cast(assignments, votes.dtype))
