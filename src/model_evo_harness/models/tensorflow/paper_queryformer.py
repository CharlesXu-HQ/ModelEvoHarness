"""Standalone, single-view QueryFormer reference components (arXiv:2609.16548v1).

Implements the §4.2 query bridge, not the competition champion's exact code.
Post-LN residuals, sigmoid gates, biased projections, and dropout=0 are explicit
engineering choices where the paper leaves details unspecified. Masks are bool,
True means valid, and masked tokens may contain nonfinite padding.
"""

import tensorflow as tf


def _check(tokens, mask, dim):
    if mask.dtype != tf.bool:
        raise TypeError("mask must have boolean dtype")
    tf.debugging.assert_rank(tokens, 3, message="tokens must have rank 3")
    tf.debugging.assert_rank(mask, 2, message="mask must have rank 2")
    tf.debugging.assert_equal(tf.shape(tokens)[:2], tf.shape(mask), message="token/mask shape mismatch")
    tf.debugging.assert_equal(tf.shape(tokens)[2], dim, message="token width mismatch")
    return tf.where(mask[..., None], tokens, tf.zeros_like(tokens))


def _mean(tokens, mask):
    clean = tf.where(mask[..., None], tokens, tf.zeros_like(tokens))
    work = tf.cast(clean, tf.float32) if tokens.dtype in (tf.float16, tf.bfloat16) else clean
    count = tf.reduce_sum(tf.cast(mask, work.dtype), axis=1, keepdims=True)
    result = tf.reduce_sum(work, axis=1) / tf.maximum(count, 1)
    return tf.cast(result, tokens.dtype)


def _max(tokens, mask):
    low = tf.cast(float("-inf"), tokens.dtype)
    clean = tf.where(mask[..., None], tokens, low)
    sentinel = tf.fill([tf.shape(tokens)[0], 1, tf.shape(tokens)[2]], low)
    result = tf.reduce_max(tf.concat([clean, sentinel], axis=1), axis=1)
    return tf.where(tf.reduce_any(mask, axis=1, keepdims=True), result, tf.zeros_like(result))


class _Attention(tf.keras.layers.Layer):
    """Scaled dot-product multihead attention with exact-zero absent outputs."""

    def __init__(self, dim, num_heads):
        super().__init__(name="attention")
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim and num_heads must be positive, with dim divisible by num_heads")
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.q_proj = tf.keras.layers.Dense(dim)
        self.k_proj = tf.keras.layers.Dense(dim)
        self.v_proj = tf.keras.layers.Dense(dim)
        self.out_proj = tf.keras.layers.Dense(dim)

    def call(self, queries, keys, query_mask, key_mask):
        queries = _check(queries, query_mask, self.dim)
        keys = _check(keys, key_mask, self.dim)
        tf.debugging.assert_equal(tf.shape(queries)[0], tf.shape(keys)[0], message="batch mismatch")
        batch, length = tf.shape(queries)[0], tf.shape(queries)[1]
        q, k, v = self.q_proj(queries), self.k_proj(keys), self.v_proj(keys)
        present = tf.reduce_any(key_mask, axis=1, keepdims=True)
        # Only keyless examples attend the sentinel, whose value is zero.
        sentinel = tf.zeros([batch, 1, self.dim], dtype=k.dtype)
        k, v = tf.concat([k, sentinel], 1), tf.concat([v, sentinel], 1)
        valid = tf.concat([key_mask, ~present], 1)

        def heads(x):
            x = tf.reshape(x, [batch, tf.shape(x)[1], self.num_heads, self.head_dim])
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.cast(x, tf.float32) if x.dtype in (tf.float16, tf.bfloat16) else x

        q, k, v = heads(q), heads(k), heads(v)
        scores = tf.matmul(q, k, transpose_b=True) * (self.head_dim ** -0.5)
        scores = tf.where(valid[:, None, None, :], scores, tf.cast(float("-inf"), scores.dtype))
        mixed = tf.matmul(tf.nn.softmax(scores, axis=-1), v)
        mixed = tf.reshape(tf.transpose(mixed, [0, 2, 1, 3]), [batch, length, self.dim])
        output = self.out_proj(tf.cast(mixed, queries.dtype))
        return tf.where((query_mask & present)[..., None], output, tf.zeros_like(output))


class CoTransformerBlock(tf.keras.layers.Layer):
    """Self-attend both sides, then let user queries attend item keys/values.

    Inputs are user [B,U,D], item [B,I,D], and their [B,U]/[B,I] masks.
    Returns updated (user, item) tokens with the same shapes. Each layer uses
    self-attention followed by post-LN residuals, then a user-only cross update:
    LN(user + sigmoid(Dense(concat(user, delta))) * delta). The gate and norm
    placement are reference choices; no reverse cross attention or FFN is added.
    """

    def __init__(self, dim, num_heads, num_layers=2):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.dim = dim
        self.user_self = [_Attention(dim, num_heads) for _ in range(num_layers)]
        self.item_self = [_Attention(dim, num_heads) for _ in range(num_layers)]
        self.user_cross = [_Attention(dim, num_heads) for _ in range(num_layers)]
        self.user_gate = [tf.keras.layers.Dense(dim) for _ in range(num_layers)]
        self.user_self_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(num_layers)]
        self.item_self_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(num_layers)]
        self.user_cross_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(num_layers)]

    def call(self, user_tokens, item_tokens, user_mask, item_mask):
        user = _check(user_tokens, user_mask, self.dim)
        item = _check(item_tokens, item_mask, self.dim)
        tf.debugging.assert_equal(tf.shape(user)[0], tf.shape(item)[0], message="batch mismatch")
        for index in range(len(self.user_self)):
            user = self.user_self_norm[index](user + self.user_self[index](user, user, user_mask, user_mask))
            item = self.item_self_norm[index](item + self.item_self[index](item, item, item_mask, item_mask))
            user = tf.where(user_mask[..., None], user, tf.zeros_like(user))
            item = tf.where(item_mask[..., None], item, tf.zeros_like(item))
            delta = self.user_cross[index](user, item, user_mask, item_mask)
            gate = tf.sigmoid(self.user_gate[index](tf.concat([user, delta], axis=-1)))
            user = self.user_cross_norm[index](user + gate * delta)
            user = tf.where(user_mask[..., None], user, tf.zeros_like(user))
        return user, item


class QuerySeqCrossAttn(tf.keras.layers.Layer):
    """Retrieve each history using the masked mean of item tokens as query.

    Inputs: item [B,I,D], S histories [B,L_s,D], and corresponding bool masks.
    Outputs: retrieved [B,S,D] and valid_domains [B,S]. Each domain has its own
    attention parameters; missing item or history evidence produces exact zeros.
    """

    def __init__(self, dim, num_heads, num_domains):
        super().__init__()
        if num_domains <= 0:
            raise ValueError("num_domains must be positive")
        self.dim, self.num_domains = dim, num_domains
        self.attentions = [_Attention(dim, num_heads) for _ in range(num_domains)]

    def call(self, item_tokens, histories, item_mask, history_masks):
        if len(histories) != self.num_domains or len(history_masks) != self.num_domains:
            raise ValueError("history and mask counts must match num_domains")
        item = _check(item_tokens, item_mask, self.dim)
        query = _mean(item, item_mask)[:, None, :]
        item_valid = tf.reduce_any(item_mask, axis=1, keepdims=True)
        retrieved, valid_domains = [], []
        for attention, history, mask in zip(self.attentions, histories, history_masks):
            history = _check(history, mask, self.dim)
            tf.debugging.assert_equal(tf.shape(item)[0], tf.shape(history)[0], message="batch mismatch")
            retrieved.append(attention(query, history, item_valid, mask)[:, 0, :])
            valid_domains.append(item_valid[:, 0] & tf.reduce_any(mask, axis=1))
        return tf.stack(retrieved, axis=1), tf.stack(valid_domains, axis=1)


class SeqQueryCrossAttn(tf.keras.layers.Layer):
    """Use each history's raw mean/max probes to attend augmented context.

    Inputs: context [B,C,D], S histories [B,L_s,D], and corresponding bool masks.
    Returns queries [B,S,2,D] and query_mask [B,S,2], with probes ordered mean,
    max. Context supplies keys and values (Eq. 6); the result has no additional
    residual or normalization. Empty history or context yields zero queries.
    """

    def __init__(self, dim, num_heads, num_domains):
        super().__init__()
        if num_domains <= 0:
            raise ValueError("num_domains must be positive")
        self.dim, self.num_domains = dim, num_domains
        self.attentions = [_Attention(dim, num_heads) for _ in range(num_domains)]

    def call(self, context_tokens, histories, context_mask, history_masks):
        if len(histories) != self.num_domains or len(history_masks) != self.num_domains:
            raise ValueError("history and mask counts must match num_domains")
        context = _check(context_tokens, context_mask, self.dim)
        context_valid = tf.reduce_any(context_mask, axis=1)
        queries, masks = [], []
        for attention, history, mask in zip(self.attentions, histories, history_masks):
            history = _check(history, mask, self.dim)
            tf.debugging.assert_equal(tf.shape(context)[0], tf.shape(history)[0], message="batch mismatch")
            probes = tf.stack([_mean(history, mask), _max(history, mask)], axis=1)
            valid = tf.reduce_any(mask, axis=1) & context_valid
            query_mask = tf.repeat(valid[:, None], 2, axis=1)
            queries.append(attention(probes, context, query_mask, context_mask))
            masks.append(query_mask)
        return tf.stack(queries, axis=1), tf.stack(masks, axis=1)


class QueryFormerQueryBridge(tf.keras.layers.Layer):
    """Single-view query bridge, excluding the later encoder/mixer/ranking head.

    Accepts user [B,U,D], item [B,I,D], S histories [B,L_s,D], their bool masks,
    and supports dynamic B/U/I/L_s including zero token lengths. Returns a dict:
    queries [B,S,2,D], query_mask [B,S,2], context_tokens [B,U+I+S,D], and
    context_mask [B,U+I+S]. Context order is updated user, updated item, retrieved
    domain tokens. Multi-view expansion, candidate packing, and training are out
    of scope. This is a clean-room reference, not exact champion code.
    """

    def __init__(self, dim, num_heads, num_domains, num_layers=2):
        super().__init__()
        self.co_transformer = CoTransformerBlock(dim, num_heads, num_layers)
        self.retrieval = QuerySeqCrossAttn(dim, num_heads, num_domains)
        self.query_generator = SeqQueryCrossAttn(dim, num_heads, num_domains)

    def call(self, user_tokens, item_tokens, histories, user_mask, item_mask, history_masks):
        user, item = self.co_transformer(user_tokens, item_tokens, user_mask, item_mask)
        retrieved, valid_domains = self.retrieval(item, histories, item_mask, history_masks)
        context = tf.concat([user, item, retrieved], axis=1)
        context_mask = tf.concat([user_mask, item_mask, valid_domains], axis=1)
        queries, query_mask = self.query_generator(context, histories, context_mask, history_masks)
        return {"queries": queries, "query_mask": query_mask,
                "context_tokens": context, "context_mask": context_mask}
