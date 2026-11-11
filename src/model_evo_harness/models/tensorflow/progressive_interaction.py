"""Initial-affinity progressive interaction, inspired by MGDIN §3.2 (Keras).

Source: https://arxiv.org/html/2603.12586v1 . This independent local block
uses raw initial dot affinities, masked softmax, and a shared token FFN;
it does not reproduce the paper's learned initial projections, unnormalized
interaction, semantic-ID tokenizer, or multiple grouping granularities.
"""

from numbers import Integral

import tensorflow as tf


class _InteractionLayer(tf.keras.layers.Layer):
    def __init__(self, dim, hidden_dim, name):
        super().__init__(name=name)
        self.scale = dim ** -0.5
        self.query = tf.keras.layers.Dense(dim, use_bias=False)
        self.key = tf.keras.layers.Dense(dim, use_bias=False)
        self.value = tf.keras.layers.Dense(dim, use_bias=False)
        self.projection = tf.keras.layers.Dense(dim, use_bias=False)
        self.ffn_in = tf.keras.layers.Dense(hidden_dim)
        self.ffn_out = tf.keras.layers.Dense(dim)
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, tokens, edge_mask):
        logits = tf.linalg.matmul(self.query(tokens), self.key(tokens), transpose_b=True) * self.scale
        logits = tf.where(edge_mask, logits, tf.cast(-float("inf"), logits.dtype))
        # A global budget can leave entire rows empty. Never softmax all -inf.
        logits = tf.where(tf.reduce_any(edge_mask, -1, keepdims=True), logits, 0)
        weights = tf.nn.softmax(logits, axis=-1) * tf.cast(edge_mask, logits.dtype)
        context = tf.linalg.matmul(weights, self.value(tokens))
        states = self.attention_norm(tokens + self.projection(context))
        return self.ffn_norm(states + self.ffn_out(tf.nn.gelu(self.ffn_in(states))))


class ProgressiveInteraction(tf.keras.Model):
    """A host-backbone block: [B,F,D] -> [B,F,D], preserving the field axis.

    ``edge_budgets`` gives one global directed-edge count per layer. Counts
    must be nondecreasing integers in [0,F²]; self-edges count normally and
    are not forced open. Ranking uses X0 @ X0.T / sqrt(D), highest first,
    with row-major tie order, once per input batch and reused across layers.
    This hard ranking is not differentiable; the current Q/K/V and FFN are.

    Optional bool ``valid_mask`` [B,F] excludes padded queries and keys
    before arithmetic, including NaN/Inf padding. Budgets clip to the number
    of valid pairs. Empty rows have zero attention context but retain their
    residual/FFN update; padded fields always return zero. Valid inputs must
    be finite floating point tensors.

    ``interaction_masks`` and ``return_masks=True`` expose bool [B,L,F,F].
    [F²]*L gives dense attention and [k]*L gives fixed sparsity with identical
    parameters: L*(4D² + 2DH + H + 5D), H = hidden_dim (default 4D).
    Computation still forms dense F×F matrices; sparse budgets do not imply
    sparse-kernel speedups. The host supplies embeddings, groups and head.
    """

    def __init__(self, num_fields: int, dim: int, edge_budgets,
                 hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = 4 * dim if hidden_dim is None else hidden_dim
        if any(not isinstance(n, Integral) or isinstance(n, bool) or n < 1
               for n in (num_fields, dim, hidden_dim)):
            raise ValueError("num_fields, dim and hidden_dim must be positive integers")
        try:
            budgets = tuple(edge_budgets)
        except TypeError as exc:
            raise ValueError("edge budgets must be a nonempty integer sequence") from exc
        if (not budgets or any(not isinstance(k, Integral) or isinstance(k, bool) or
                               k < 0 or k > num_fields ** 2 for k in budgets) or
                any(a > b for a, b in zip(budgets, budgets[1:]))):
            raise ValueError("edge budgets must be nondecreasing integers in [0,num_fields²]")
        self.num_fields, self.dim = num_fields, dim
        self.edge_budgets = budgets
        self.blocks = [_InteractionLayer(dim, hidden_dim, name=f"interaction_{i}")
                       for i in range(len(budgets))]

    def _prepare(self, tokens, valid_mask):
        tokens = tf.ensure_shape(tokens, (None, self.num_fields, self.dim))
        if not tokens.dtype.is_floating:
            raise ValueError("tokens must have a floating point dtype")
        if valid_mask is None:
            valid_mask = tf.ones(tf.shape(tokens)[:2], tf.bool)
        else:
            if valid_mask.dtype != tf.bool:
                raise ValueError("valid_mask must be bool [B,num_fields]")
            valid_mask = tf.ensure_shape(valid_mask, (None, self.num_fields))
            tf.debugging.assert_equal(tf.shape(valid_mask)[0], tf.shape(tokens)[0])
        safe = tf.where(valid_mask[..., None], tokens, 0)
        safe = tf.debugging.check_numerics(safe, "valid tokens must be finite")
        return safe, valid_mask

    def _masks(self, tokens, valid_mask):
        initial = tf.stop_gradient(tokens)
        affinities = tf.linalg.matmul(initial, initial, transpose_b=True) * self.dim ** -0.5
        pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
        flat = tf.reshape(tf.where(pairs, affinities, tf.cast(-float("inf"), affinities.dtype)),
                          (-1, self.num_fields ** 2))
        order = tf.argsort(flat, axis=-1, direction="DESCENDING", stable=True)
        ranks = tf.reshape(tf.argsort(order, axis=-1), tf.shape(affinities))
        return tf.stack([pairs & (ranks < k) for k in self.edge_budgets], axis=1)

    def interaction_masks(self, tokens: tf.Tensor,
                          valid_mask: tf.Tensor | None = None) -> tf.Tensor:
        """Return the initial-ranking masks [B,L,F,F] without updating tokens."""
        tokens, valid_mask = self._prepare(tokens, valid_mask)
        return self._masks(tokens, valid_mask)

    def call(self, tokens: tf.Tensor, valid_mask: tf.Tensor | None = None,
             return_masks: bool = False):
        states, valid_mask = self._prepare(tokens, valid_mask)
        masks = self._masks(states, valid_mask)
        for i, block in enumerate(self.blocks):
            states = block(states, masks[:, i])
            states = tf.where(valid_mask[..., None], states, 0)
        return (states, masks) if return_masks else states
