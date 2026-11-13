"""Independent TensorFlow SIM/ETA/SDIM/TWIN/MIRRN tensor mechanisms.

Reference: reczoo/FuxiCTR, commit 183da1c944ba8930fcdcd54114837c081959d748,
model_zoo/LongCTR/{SIM,ETA,SDIM,TWIN,MIRRN}. Host owns embedding construction,
time-safe histories, short branches, heads, objectives and evaluation. Dense
hash scans are not an ANN index. Masks use True=valid. Invalid K slots have
index -1, False mask and zero vectors; K may exceed the valid history length.
"""
import math

import tensorflow as tf


def _dense(dim, output_dim=None):
    layer = tf.keras.layers.Dense(output_dim or dim, use_bias=False)
    layer.build((None, dim))
    return layer


def _inputs(target, history, valid, dim):
    if valid.dtype != tf.bool:
        raise ValueError("valid must be boolean")
    tf.debugging.assert_rank(target, 2)
    tf.debugging.assert_rank(history, 3)
    tf.debugging.assert_rank(valid, 2)
    tf.debugging.assert_equal(tf.shape(target), [tf.shape(history)[0], dim])
    tf.debugging.assert_equal(tf.shape(history)[-1], dim)
    tf.debugging.assert_equal(tf.shape(valid), tf.shape(history)[:2])
    return tf.where(valid[..., None], history, tf.zeros_like(history))


def _weights(scores, valid):
    safe = tf.where(valid, scores, tf.cast(float("-inf"), scores.dtype))
    dummy = tf.where(tf.reduce_any(valid, -1, keepdims=True), tf.cast(float("-inf"), scores.dtype), tf.cast(0., scores.dtype))
    return tf.nn.softmax(tf.concat([safe, dummy], -1), -1)[..., :-1] * tf.cast(valid, scores.dtype)


def _mean(history, valid):
    return tf.reduce_sum(history, 1) / tf.maximum(tf.reduce_sum(tf.cast(valid, history.dtype), 1, keepdims=True), 1.)


def _select(history, valid, scores, k, chronological=False):
    b, length, dim = tf.shape(history)[0], tf.shape(history)[1], tf.shape(history)[2]
    values = tf.concat([history, tf.zeros([b, k, dim], history.dtype)], 1)
    mask = tf.concat([valid, tf.zeros([b, k], tf.bool)], 1)
    score = tf.concat([tf.where(valid, scores, tf.cast(float("-inf"), scores.dtype)), tf.fill([b, k], tf.cast(float("-inf"), scores.dtype))], 1)
    index = tf.argsort(score, axis=-1, direction="DESCENDING", stable=True)[:, :k]
    selected_valid = tf.gather(mask, index, axis=1, batch_dims=1)
    if chronological:
        order = tf.argsort(tf.where(selected_valid, index, length+k), axis=-1, stable=True)
        index = tf.gather(index, order, axis=1, batch_dims=1)
        selected_valid = tf.gather(selected_valid, order, axis=1, batch_dims=1)
    selected = tf.gather(values, index, axis=1, batch_dims=1)
    selected = tf.where(selected_valid[..., None], selected, tf.zeros_like(selected))
    return selected, selected_valid, tf.where(selected_valid, index, -1)


class _TargetAttention(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__(name="targetattention")
        self.query, self.key, self.value = _dense(dim), _dense(dim), _dense(dim)
        self.scale = math.sqrt(dim)

    def call(self, target, history, valid):
        scores = tf.einsum("bd,bld->bl", self.query(target), self.key(history)) / self.scale
        attention = _weights(scores, valid)
        return tf.einsum("bl,bld->bd", attention, self.value(history)), attention


class SIMHistorySelector(tf.keras.Model):
    """Unnormalized learned coarse sum plus hard top-K learned fine attention.

    Dict: coarse/fine/representation [B,D], coarse_scores [B,L], selected
    [B,K,D], indices/selected_mask/attention [B,K]. Negative scores remain in
    coarse sum; invalid events cannot displace negative valid scores. Host
    supplies coarse/fine heads and losses; hard indices are not differentiable.
    """
    def __init__(self, dim, topk=5):
        super().__init__()
        if min(dim, topk) < 1:
            raise ValueError("dim and topk must be positive")
        self.dim, self.topk = dim, topk
        self.query, self.key = _dense(dim), _dense(dim)
        self.fine_attention = _TargetAttention(dim)

    def call(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim)
        scores = tf.einsum("bd,bld->bl", self.query(target), self.key(history))
        scores = tf.where(valid, scores, tf.zeros_like(scores))
        coarse = tf.einsum("bl,bld->bd", scores, history)
        selected, mask, indices = _select(history, valid, scores, self.topk)
        fine, attention = self.fine_attention(target, selected, mask)
        return dict(representation=fine, coarse=coarse, fine=fine, coarse_scores=scores,
                    selected=selected, selected_mask=mask, indices=indices, attention=attention)


class ETAHistorySelector(tf.keras.Model):
    """Checkpointed fixed random-hyperplane Hamming top-K then target attention.

    Dict: representation [B,D], selected [B,K,D], selected_mask/indices/attention
    [B,K], hash_distance [B,L] (invalid=0). Ties use earlier original indices.
    Projection <0 defines a bit; no stochastic rehashing or ANN cache.
    """
    def __init__(self, dim, topk=5, hash_bits=16):
        super().__init__()
        if min(dim, topk, hash_bits) < 1:
            raise ValueError("dimensions and topk must be positive")
        self.dim, self.topk = dim, topk
        self.rotations = self.add_weight(name="rotations", shape=(dim, hash_bits), initializer="random_normal", trainable=False)
        self.attention = _TargetAttention(dim)

    def call(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim)
        distance = tf.reduce_sum(tf.cast((tf.matmul(history, self.rotations) < 0) != (target @ self.rotations < 0)[:, None], tf.int32), -1)
        selected, mask, indices = _select(history, valid, -tf.cast(distance, history.dtype), self.topk)
        representation, attention = self.attention(target, selected, mask)
        return dict(representation=representation, selected=selected, selected_mask=mask,
                    indices=indices, hash_distance=tf.where(valid, distance, 0), attention=attention)


class SDIMCollisionPooling(tf.keras.Model):
    """Sum exact bucket collisions per hash, average over hashes; no L2 norm.

    Dict: collision_mask [B,H,L], per_hash [B,H,D], representation [B,D].
    No trainable parameters; history embeddings receive gradients, while hash
    membership and target queries do not. No attention weights are inferred.
    """
    def __init__(self, dim, num_hashes=2, hash_bits=4):
        super().__init__()
        if min(dim, num_hashes, hash_bits) < 1:
            raise ValueError("dimensions and hash counts must be positive")
        self.dim = dim
        self.rotations = self.add_weight(name="rotations", shape=(num_hashes, dim, hash_bits), initializer="random_normal", trainable=False)

    def call(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim)
        history_bits = tf.einsum("bld,hdf->bhlf", history, self.rotations) < 0
        target_bits = tf.einsum("bd,hdf->bhf", target, self.rotations) < 0
        collision = tf.reduce_all(history_bits == target_bits[:, :, None], -1) & valid[:, None]
        per_hash = tf.einsum("bhl,bld->bhd", tf.cast(collision, history.dtype), history)
        return dict(representation=tf.reduce_mean(per_hash, 1), per_hash=per_hash, collision_mask=collision)


class TWINHeadwiseSelector(tf.keras.Model):
    """Head-specific learned top-K and fine normalization share q/k scores.

    Dict: selected [B,H,K,D/H], indices/selected_mask/attention [B,H,K],
    heads [B,H,D/H], representation [B,D]. Dense search, no ANN cache.
    """
    def __init__(self, dim, num_heads=2, topk=5):
        super().__init__()
        if min(dim, num_heads, topk) < 1 or dim % num_heads:
            raise ValueError("positive dimensions and dim divisible by heads required")
        self.dim, self.num_heads, self.topk = dim, num_heads, topk
        self.query, self.key, self.value, self.out_proj = [_dense(dim) for _ in range(4)]

    def call(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim)
        b, length = tf.shape(history)[0], tf.shape(history)[1]
        h, d = self.num_heads, self.dim // self.num_heads
        query = tf.reshape(self.query(target), [b, h, d])
        key = tf.transpose(tf.reshape(self.key(history), [b, length, h, d]), [0, 2, 1, 3])
        value = tf.transpose(tf.reshape(self.value(history), [b, length, h, d]), [0, 2, 1, 3])
        scores = tf.einsum("bhd,bhld->bhl", query, key) / math.sqrt(d)
        mask = tf.reshape(tf.broadcast_to(valid[:, None], [b, h, length]), [b*h, length])
        selected, selected_mask, indices = _select(tf.reshape(value, [b*h, length, d]), mask,
                                                   tf.reshape(scores, [b*h, length]), self.topk)
        padded_scores = tf.concat([tf.reshape(scores, [b*h, length]), tf.zeros([b*h, 1], scores.dtype)], -1)
        chosen = tf.gather(padded_scores, tf.where(selected_mask, indices, length), axis=1, batch_dims=1)
        attention = _weights(chosen, selected_mask)
        heads = tf.reshape(tf.reduce_sum(attention[..., None] * selected, 1), [b, h, d])
        return dict(representation=self.out_proj(tf.reshape(heads, [b, self.dim])), heads=heads,
                    selected=tf.reshape(selected, [b, h, self.topk, d]),
                    selected_mask=tf.reshape(selected_mask, [b, h, self.topk]),
                    indices=tf.reshape(indices, [b, h, self.topk]), attention=tf.reshape(attention, [b, h, self.topk]))


class MIRRNHistoryRefiner(tf.keras.Model):
    """Target/recent/global LSH retrieval and masked complex channel refinement.

    recent_mask [B,L] must be a subset of valid. Selected original indices are
    restored to chronological order. rFFT/irFFT has explicit n=K and orthonormal
    scaling, including odd K. Full block-channel mixing replaces the source's
    diagonal einsum; masks apply before FFT and after residual LayerNorm. The
    matrices are frequency-shared; zero padding is part of this K-point signal.
    Omits source recency-position additions and dropout, deliberately.
    Dict: representation [B,D], interests [B,3,D], interest_mask/attention
    [B,3], refined/selected [B,3,K,D], indices/selected_mask [B,3,K].
    """
    def __init__(self, dim, topk=5, hash_bits=16, num_blocks=2):
        super().__init__()
        if min(dim, topk, hash_bits, num_blocks) < 1 or dim % num_blocks:
            raise ValueError("positive sizes and dim divisible by blocks required")
        self.dim, self.topk, self.num_blocks = dim, topk, num_blocks
        self.rotations = self.add_weight(name="rotations", shape=(dim, hash_bits), initializer="random_normal", trainable=False)
        width = dim // num_blocks
        self.frequency_real = self.add_weight(name="frequency_real", shape=(3, num_blocks, width, width), initializer=tf.keras.initializers.RandomNormal(stddev=.02))
        self.frequency_imag = self.add_weight(name="frequency_imag", shape=(3, num_blocks, width, width), initializer=tf.keras.initializers.RandomNormal(stddev=.02))
        self.norms = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(3)]
        for norm in self.norms:
            norm.build((None, None, dim))
        self.attention = _TargetAttention(dim)

    def refine(self, selected, selected_mask, branch):
        values = tf.where(selected_mask[..., None], selected, tf.zeros_like(selected))
        b, length = tf.shape(values)[0], tf.shape(values)[1]
        scale = tf.sqrt(tf.cast(length, values.dtype))
        # TensorFlow transforms the final axis; transpose time there explicitly.
        spectrum = tf.transpose(tf.signal.rfft(tf.transpose(values, [0, 2, 1])), [0, 2, 1]) / tf.cast(scale, tf.complex64)
        spectrum = tf.reshape(spectrum, [b, length//2+1, self.num_blocks, self.dim//self.num_blocks])
        matrix = tf.complex(self.frequency_real[branch], self.frequency_imag[branch])
        mixed = tf.einsum("bfni,nij->bfnj", spectrum, matrix)
        mixed = tf.reshape(mixed, [b, length//2+1, self.dim])
        filtered = tf.transpose(tf.signal.irfft(tf.transpose(mixed, [0, 2, 1]), fft_length=[length]), [0, 2, 1]) * scale
        return tf.where(selected_mask[..., None], self.norms[branch](values + filtered), tf.zeros_like(values))

    def call(self, target, history, valid, recent_mask):
        history = _inputs(target, history, valid, self.dim)
        if recent_mask.dtype != tf.bool:
            raise ValueError("recent_mask must be boolean")
        tf.debugging.assert_equal(tf.shape(recent_mask), tf.shape(valid))
        tf.debugging.assert_equal(tf.reduce_any(recent_mask & ~valid), False, message="recent_mask must be a subset of valid")
        queries = [target, _mean(tf.where(recent_mask[..., None], history, tf.zeros_like(history)), recent_mask), _mean(history, valid)]
        availability = [tf.reduce_any(valid, 1), tf.reduce_any(recent_mask, 1), tf.reduce_any(valid, 1)]
        history_hash = tf.matmul(history, self.rotations) > 0
        selected_all, masks, indices_all, refined_all, interests = [], [], [], [], []
        for i, query in enumerate(queries):
            score = -tf.reduce_sum(tf.cast(history_hash != (query @ self.rotations > 0)[:, None], history.dtype), -1)
            selected, mask, indices = _select(history, valid & availability[i][:, None], score,
                                              self.topk, chronological=True)
            refined = self.refine(selected, mask, i)
            selected_all.append(selected); masks.append(mask); indices_all.append(indices)
            refined_all.append(refined); interests.append(_mean(refined, mask))
        interests = tf.stack(interests, 1)
        interest_mask = tf.stack(availability, 1)
        representation, attention = self.attention(target, interests, interest_mask)
        return dict(representation=representation, interests=interests, interest_mask=interest_mask,
                    refined=tf.stack(refined_all, 1), selected=tf.stack(selected_all, 1),
                    selected_mask=tf.stack(masks, 1), indices=tf.stack(indices_all, 1), attention=attention)
