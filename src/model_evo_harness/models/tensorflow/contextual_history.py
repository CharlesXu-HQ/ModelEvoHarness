"""Original local components inspired by context-rich SIF events and CCFormer reads.

Sources: https://arxiv.org/html/2604.15650v3 (§3.1) and
https://arxiv.org/html/2607.28070v2 (§3.1). This implements neither a SIF
quantizer nor a full CCFormer hierarchy. Only TensorFlow is required.
All masks are bool, True=valid. Valid features must be finite floating tensors;
masked NaN/Inf padding is sanitized before projection and has zero gradient.
"""

import tensorflow as tf


def _check(tokens, mask, dim):
    if mask.dtype != tf.bool:
        raise TypeError("mask must be bool with True=valid")
    if not tokens.dtype.is_floating:
        raise TypeError("tokens must have floating dtype")
    tf.debugging.assert_rank(tokens, 3, message="tokens must have rank 3")
    tf.debugging.assert_rank(mask, 2, message="mask must have rank 2")
    tf.debugging.assert_equal(tf.shape(tokens)[:2], tf.shape(mask), message="token/mask shape mismatch")
    tf.debugging.assert_equal(tf.shape(tokens)[2], dim, message="token width mismatch")
    return tf.where(mask[..., None], tokens, tf.zeros_like(tokens))


class ContextualHistoryEncoder(tf.keras.layers.Layer):
    """Aligned item [B,L,I], context [B,L,C], mask [B,L] -> tokens, mask.

    tokens[b,t] = tanh(W_item item[b,t] + W_context context[b,t] + bias).
    Output width is dim. Every event keeps its input position; no pooling,
    sorting, positional encoding or history creation is performed. Supply time
    or position in context when temporal distinctions are needed downstream.
    Context fields must be aligned to each event and known before prediction.
    The affine/tanh fusion is an engineering choice, not SIF tokenization.
    """

    def __init__(self, item_dim: int, context_dim: int, dim: int):
        super().__init__()
        if min(item_dim, context_dim, dim) < 1:
            raise ValueError("item_dim, context_dim and dim must be positive")
        self.item_dim, self.context_dim = item_dim, context_dim
        self.item_proj = tf.keras.layers.Dense(dim)
        self.context_proj = tf.keras.layers.Dense(dim, use_bias=False)

    def call(self, item_features, context_features, history_mask):
        items = _check(item_features, history_mask, self.item_dim)
        context = _check(context_features, history_mask, self.context_dim)
        tokens = tf.tanh(self.item_proj(items) + self.context_proj(context))
        return tf.where(history_mask[..., None], tokens, tf.zeros_like(tokens)), history_mask


class _RelationAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, name):
        super().__init__(name=name)
        self.dim, self.num_heads = dim, num_heads
        self.q_proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.k_proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.v_proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.out_proj = tf.keras.layers.Dense(dim, use_bias=False)

    def call(self, queries, keys, query_mask, key_mask):
        batch, length = tf.shape(queries)[0], tf.shape(queries)[1]
        width = self.dim // self.num_heads
        q, k, v = self.q_proj(queries), self.k_proj(keys), self.v_proj(keys)
        # A zero sentinel is visible only when all real keys are absent.
        present = tf.reduce_any(key_mask, axis=1, keepdims=True)
        valid_keys = tf.concat([key_mask, ~present], 1)
        sentinel = tf.zeros([batch, 1, self.dim], k.dtype)
        k, v = tf.concat([k, sentinel], 1), tf.concat([v, sentinel], 1)

        def heads(x):
            split = tf.transpose(tf.reshape(x, [batch, tf.shape(x)[1], self.num_heads, width]), [0, 2, 1, 3])
            return tf.cast(split, tf.float32) if split.dtype in (tf.float16, tf.bfloat16) else split

        scores = tf.matmul(heads(q), heads(k), transpose_b=True) * (width ** -0.5)
        scores = tf.where(valid_keys[:, None, None, :], scores, tf.cast(float("-inf"), scores.dtype))
        weights = tf.nn.softmax(scores, axis=-1)
        mixed = tf.matmul(weights, heads(v))
        mixed = tf.reshape(tf.transpose(mixed, [0, 2, 1, 3]), [batch, length, self.dim])
        output = self.out_proj(tf.cast(mixed, q.dtype))
        valid = query_mask & present
        output = tf.where(valid[..., None], output, tf.zeros_like(output))
        inspected = tf.where(valid[:, None, :, None], weights[..., :-1], tf.zeros_like(weights[..., :-1]))
        return output, valid, inspected


class RelationHistoryAttention(tf.keras.layers.Layer):
    """Independent user->history, target->history, target->user read branches.

    Inputs: user [B,U,D], history [B,L,D], target [B,T,D] and corresponding
    bool masks [B,U], [B,L], [B,T]. Outputs are a dict with user_history
    [B,U,D], target_history [B,T,D], target_user [B,T,D] and each *_mask.
    return_attention=True adds an attention dict keyed by the same relation
    names, with [B,H,U,L], [B,H,T,L], [B,H,T,U] weights, respectively.
    Each valid query sums to one over its own available keys; absent evidence
    or masked queries return exact-zero representations and weights.

    Relations own separate bias-free Q/K/V/output projections. User and history
    inputs remain unchanged, and target->user reads the original user tokens.
    Targets never attend each other, so separate candidate calls and batching
    agree. "Fixed evidence" does not mean detached gradients: callers control
    freezing/detachment. No residual, FFN, fusion, compression or head is added;
    callers may concatenate these branches with an existing backbone state.
    """

    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        if dim < 1 or num_heads < 1 or dim % num_heads:
            raise ValueError("positive dim must be divisible by positive num_heads")
        self.dim = dim
        self.user_history = _RelationAttention(dim, num_heads, "user_history")
        self.target_history = _RelationAttention(dim, num_heads, "target_history")
        self.target_user = _RelationAttention(dim, num_heads, "target_user")

    def call(self, user_tokens, history_tokens, target_tokens, user_mask,
             history_mask, target_mask, return_attention=False):
        users = _check(user_tokens, user_mask, self.dim)
        history = _check(history_tokens, history_mask, self.dim)
        targets = _check(target_tokens, target_mask, self.dim)
        tf.debugging.assert_equal(tf.shape(users)[0], tf.shape(history)[0], message="user/history batch mismatch")
        tf.debugging.assert_equal(tf.shape(users)[0], tf.shape(targets)[0], message="user/target batch mismatch")
        result, attention = {}, {}
        for name, queries, keys, qm, km in (
            ("user_history", users, history, user_mask, history_mask),
            ("target_history", targets, history, target_mask, history_mask),
            ("target_user", targets, users, target_mask, user_mask),
        ):
            result[name], result[name + "_mask"], attention[name] = getattr(self, name)(queries, keys, qm, km)
        if return_attention:
            result["attention"] = attention
        return result
