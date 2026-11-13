"""Independent TensorFlow BST/DMIN/DMR tensor references.

Source mechanisms: reczoo/FuxiCTR 183da1c944ba8930fcdcd54114837c081959d748,
model_zoo/{BST,DMIN,DMR}/src. Host owns time-safe, aligned item/context
embeddings, prediction heads, labels, objectives and evaluation. True=valid;
masked NaNs are sanitized. Empty/all-masked history readouts are zero.
"""
import math

import tensorflow as tf


def _dense(input_dim, output_dim, bias=True):
    layer = tf.keras.layers.Dense(output_dim, use_bias=bias)
    layer.build((None, input_dim))
    return layer


def _norm(dim):
    layer = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    layer.build((None, None, dim))
    return layer


def _inputs(target, history, valid, dim, max_len):
    if valid.dtype != tf.bool:
        raise ValueError("valid must be boolean")
    tf.debugging.assert_rank(target, 2)
    tf.debugging.assert_rank(history, 3)
    tf.debugging.assert_rank(valid, 2)
    tf.debugging.assert_equal(tf.shape(target), [tf.shape(history)[0], dim])
    tf.debugging.assert_equal(tf.shape(history)[-1], dim)
    tf.debugging.assert_equal(tf.shape(valid), tf.shape(history)[:2])
    tf.debugging.assert_less_equal(tf.shape(history)[1], max_len)
    return tf.where(valid[..., None], history, tf.zeros_like(history))


def _weights(scores, valid):
    valid = tf.broadcast_to(valid, tf.shape(scores))
    masked = tf.where(valid, scores, tf.cast(float("-inf"), scores.dtype))
    dummy = tf.where(tf.reduce_any(valid, -1, keepdims=True), tf.cast(float("-inf"), scores.dtype), tf.cast(0., scores.dtype))
    return tf.nn.softmax(tf.concat([masked, dummy], -1), -1)[..., :-1] * tf.cast(valid, scores.dtype)


def _last(values, valid):
    b, length, dim = tf.shape(values)[0], tf.shape(values)[1], tf.shape(values)[2]
    index = tf.where(valid, tf.range(length)[None] + 1, 0)
    index = tf.reduce_max(tf.concat([index, tf.zeros([b, 1], tf.int32)], 1), 1)
    padded = tf.concat([tf.zeros([b, 1, dim], values.dtype), values], 1)
    return tf.gather(padded, index, axis=1, batch_dims=1)


class _SelfHeads(tf.keras.layers.Layer):
    def __init__(self, dim, heads):
        super().__init__(name="selfheads")
        self.dim, self.heads = dim, heads
        self.query, self.key, self.value = [_dense(dim, dim, False) for _ in range(3)]

    def call(self, sequence, valid, causal=False):
        b, length = tf.shape(sequence)[0], tf.shape(sequence)[1]
        shape = [b, length, self.heads, self.dim//self.heads]
        query, key, value = [tf.transpose(tf.reshape(layer(sequence), shape), [0, 2, 1, 3])
                             for layer in (self.query, self.key, self.value)]
        scores = tf.matmul(query, key, transpose_b=True) / math.sqrt(self.dim//self.heads)
        allowed = valid[:, None, None, :] & valid[:, None, :, None]
        if causal:
            allowed = allowed & tf.linalg.band_part(tf.ones([length, length], tf.bool), -1, 0)
        weights = _weights(scores, allowed)
        return tf.matmul(weights, value)


class _Refiner(tf.keras.layers.Layer):
    def __init__(self, dim, heads):
        super().__init__(name="refiner")
        self.attention = _SelfHeads(dim, heads)
        self.out_proj = _dense(dim, dim, False)
        self.norm1, self.norm2 = _norm(dim), _norm(dim)
        self.ff1, self.ff2 = _dense(dim, 2*dim), _dense(2*dim, dim)

    def call(self, values, valid, causal=False):
        b, length, dim = tf.shape(values)[0], tf.shape(values)[1], tf.shape(values)[2]
        heads = self.attention(values, valid, causal=causal)
        attended = tf.reshape(tf.transpose(heads, [0, 2, 1, 3]), [b, length, dim])
        states = self.norm1(values + self.out_proj(attended))
        states = self.norm2(states + self.ff2(tf.nn.relu(self.ff1(states))))
        return tf.where(valid[..., None], states, tf.zeros_like(states))


class BSTHistoryEncoder(tf.keras.Model):
    """Candidate appended to history, bidirectional Transformer, target readout.

    Dict: states [B,L,D], mask [B,L], candidate_state/representation [B,D],
    has_history [B]. One block with learned positions; excludes source's other
    pooling choices, causal option and prediction head. No-history readout is
    zero; a target-only branch belongs to the host.
    """
    def __init__(self, dim, num_heads=2, max_len=128):
        super().__init__()
        if min(dim, num_heads, max_len) < 1 or dim % num_heads:
            raise ValueError("positive sizes and dim divisible by heads required")
        self.dim, self.max_len = dim, max_len
        self.position = self.add_weight(name="position", shape=(max_len+1, dim), initializer=tf.keras.initializers.RandomNormal(stddev=.02))
        self.refiner = _Refiner(dim, num_heads)

    def call(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim, self.max_len)
        length = tf.shape(history)[1]
        values = tf.concat([history, target[:, None]], 1) + self.position[:length+1]
        mask = tf.concat([valid, tf.ones([tf.shape(valid)[0], 1], tf.bool)], 1)
        values = tf.where(mask[..., None], values, tf.zeros_like(values))
        states = self.refiner(values, mask, causal=False)
        has_history = tf.reduce_any(valid, 1)
        representation = tf.where(has_history[:, None], states[:, -1], tf.zeros_like(target))
        return dict(representation=representation, candidate_state=representation,
                    states=states[:, :-1], mask=valid, has_history=has_history)


class DMINInterestRefiner(tf.keras.Model):
    """Causal refiner, distinct self-attention heads and DIN-style target readout.

    Each extraction head expands to D with its own norm/FFN and target+position
    scoring. Dict: refined [B,L,D], head_states [B,H,L,D], interests [B,H,D],
    attention [B,H,L], representation [B,H*D], mask [B,L]. Independent two-norm
    refiner, no dropout; host owns negative sequences and auxiliary objectives.
    """
    def __init__(self, dim, num_heads=2, max_len=128):
        super().__init__()
        if min(dim, num_heads, max_len) < 1 or dim % num_heads:
            raise ValueError("positive sizes and dim divisible by heads required")
        self.dim, self.num_heads, self.max_len = dim, num_heads, max_len
        self.refiner = _Refiner(dim, num_heads)
        self.extractor = _SelfHeads(dim, num_heads)
        self.position = self.add_weight(name="position", shape=(max_len, dim), initializer=tf.keras.initializers.RandomNormal(stddev=.02))
        self.expansions = [_dense(dim//num_heads, dim, False) for _ in range(num_heads)]
        self.norms = [_norm(dim) for _ in range(num_heads)]
        self.ff1 = [_dense(dim, 2*dim) for _ in range(num_heads)]
        self.ff2 = [_dense(2*dim, dim) for _ in range(num_heads)]
        self.target_query = [_dense(2*dim, dim) for _ in range(num_heads)]
        self.score1 = [_dense(4*dim, dim) for _ in range(num_heads)]
        self.score2 = [_dense(dim, 1) for _ in range(num_heads)]

    def call(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim, self.max_len)
        b, length = tf.shape(history)[0], tf.shape(history)[1]
        refined = self.refiner(history, valid, causal=True)
        heads = self.extractor(refined, valid, causal=True)
        context = tf.concat([tf.broadcast_to(target[:, None], [b, length, self.dim]),
                             tf.broadcast_to(self.position[:length], [b, length, self.dim])], -1)
        interests, states, attention = [], [], []
        for i in range(self.num_heads):
            head = self.norms[i](refined + self.expansions[i](heads[:, i]))
            head = head + self.ff2[i](tf.nn.relu(self.ff1[i](head)))
            head = tf.where(valid[..., None], head, tf.zeros_like(head))
            query = self.target_query[i](context)
            joined = tf.concat([query, head, query-head, query*head], -1)
            score = tf.squeeze(self.score2[i](tf.nn.relu(self.score1[i](joined))), -1)
            weight = _weights(score, valid)
            interests.append(tf.einsum("bl,bld->bd", weight, head))
            states.append(head); attention.append(weight)
        interests = tf.stack(interests, 1)
        return dict(representation=tf.reshape(interests, [b, self.num_heads*self.dim]), interests=interests,
                    refined=refined, head_states=tf.stack(states, 1),
                    attention=tf.stack(attention, 1), mask=valid)


class DMRHistoryMatching(tf.keras.Model):
    """Context-to-history causal user states and target-to-history item matching.

    context [B,L,D] is optional aligned side information, omitted as zeros.
    User states are target-independent and matched by a target dot product.
    Item path exposes both softmax readout and sum of raw relevance scores.
    Dict: representation [B,2D], user_match/item_match [B,D], user_states
    [B,L,D], user_score/item_score [B], item_scores/attention [B,L], mask [B,L].
    Separate item input/output tables and next-event supervision are host-owned.
    """
    def __init__(self, dim, max_len=128):
        super().__init__()
        if min(dim, max_len) < 1:
            raise ValueError("dim and max_len must be positive")
        self.dim, self.max_len = dim, max_len
        self.position = self.add_weight(name="position", shape=(max_len, dim), initializer=tf.keras.initializers.RandomNormal(stddev=.02))
        self.user_query = _dense(2*dim, dim)
        self.item_query = _dense(3*dim, dim)
        self.user_score1, self.user_score2 = _dense(4*dim, dim), _dense(dim, 1)
        self.item_score1, self.item_score2 = _dense(4*dim, dim), _dense(dim, 1)
        self.user_output = _dense(dim, dim)

    def call(self, target, history, valid, context=None):
        history = _inputs(target, history, valid, self.dim, self.max_len)
        b, length = tf.shape(history)[0], tf.shape(history)[1]
        if context is None:
            context = tf.zeros_like(history)
        tf.debugging.assert_equal(tf.shape(context), tf.shape(history))
        context = tf.where(valid[..., None], context, tf.zeros_like(context))
        context = tf.concat([context, tf.broadcast_to(self.position[:length], [b, length, self.dim])], -1)
        user_query = tf.nn.relu(self.user_query(context))
        user_join = tf.concat([user_query, history, user_query-history, user_query*history], -1)
        user_scores = tf.squeeze(self.user_score2(tf.nn.relu(self.user_score1(user_join))), -1)
        allowed = valid[:, :, None] & valid[:, None, :] & tf.linalg.band_part(tf.ones([length, length], tf.bool), -1, 0)
        weights = _weights(tf.broadcast_to(user_scores[:, None], [b, length, length]), allowed)
        user_states = tf.nn.relu(self.user_output(tf.matmul(weights, history)))
        user_states = tf.where(valid[..., None], user_states, tf.zeros_like(user_states))
        user_match = _last(user_states, valid)
        item_query = tf.nn.relu(self.item_query(tf.concat([tf.broadcast_to(target[:, None], [b, length, self.dim]), context], -1)))
        item_join = tf.concat([item_query, history, item_query-history, item_query*history], -1)
        item_scores = tf.squeeze(self.item_score2(tf.nn.relu(self.item_score1(item_join))), -1)
        item_scores = tf.where(valid, item_scores, tf.zeros_like(item_scores))
        attention = _weights(item_scores, valid)
        item_match = tf.einsum("bl,bld->bd", attention, history)
        return dict(representation=tf.concat([user_match, item_match], -1), user_match=user_match,
                    item_match=item_match, user_states=user_states, user_score=tf.reduce_sum(user_match*target, -1),
                    item_score=tf.reduce_sum(item_scores, -1), item_scores=item_scores, attention=attention, mask=valid)
