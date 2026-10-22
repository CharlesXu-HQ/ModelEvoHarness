"""RankMixer dense block and UniDot cross-layer interaction readout (Keras).

RankMixer is independently implemented from arXiv:2507.15551v3, Eq. 1, 3-8.
Highway adapts FMHighway and per-domain dots from linrongc/kdd26_unidot,
commit 1399702006a1fc2f86329c8c6133f3f3ac1e6d0d, unified_model.py.
Changes: TensorFlow port, stacked API, optional domain bundle, validation,
no classifier or bus dependencies. The paper's aggregated Gram is omitted.

MIT License (UniDot portions)
Copyright (c) 2026 Rongcheng Lin and the UniDot authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf


def _token_glorot(shape, dtype=None):
    return tf.stack([tf.keras.initializers.GlorotUniform()(shape[1:], dtype=dtype)
                     for _ in range(shape[0])])


class RankMixerBlock(tf.keras.Model):
    """[B,T,D] -> [B,T,D]; fixed, fully defined field tokens, D divisible by T.

    LN(x + rewire(x)), then LN(s + independent_per_token_GELU_FFN(s)).
    No tokenizer, head, sparse experts or sequence handling.
    """

    def __init__(self, num_tokens: int, dim: int, hidden_dim: int):
        super().__init__()
        if min(num_tokens, dim, hidden_dim) < 1:
            raise ValueError("dimensions must be positive")
        if dim % num_tokens:
            raise ValueError("dim must be divisible by num_tokens")
        self.num_tokens, self.dim = num_tokens, dim
        self.w1 = self.add_weight(name="w1", shape=(num_tokens, dim, hidden_dim), initializer=_token_glorot)
        self.b1 = self.add_weight(name="b1", shape=(num_tokens, hidden_dim), initializer="zeros")
        self.w2 = self.add_weight(name="w2", shape=(num_tokens, hidden_dim, dim), initializer=_token_glorot)
        self.b2 = self.add_weight(name="b2", shape=(num_tokens, dim), initializer="zeros")
        self.mix_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def mix_tokens(self, tokens: tf.Tensor) -> tf.Tensor:
        """Exchange token/head axes; this permutation alone is self-inverse."""
        tokens = tf.ensure_shape(tokens, (None, self.num_tokens, self.dim))
        heads = tf.reshape(tokens, (-1, self.num_tokens, self.num_tokens, self.dim // self.num_tokens))
        return tf.reshape(tf.transpose(heads, (0, 2, 1, 3)), tf.shape(tokens))

    def call(self, tokens: tf.Tensor) -> tf.Tensor:
        states = self.mix_norm(tokens + self.mix_tokens(tokens))
        hidden = tf.nn.gelu(tf.einsum("btd,tdh->bth", states, self.w1) + self.b1)
        delta = tf.einsum("bth,thd->btd", hidden, self.w2) + self.b2
        return self.ffn_norm(states + delta)


class ExplicitInteractionHighway(tf.keras.Model):
    """Read out all K layers into [B,K*S*I + K*U*I], domains first.

    user_states [B,K,U,D] and item_states [B,K,I,D] are post-fusion states.
    Optional domain_dots [B,K,S,I] come from pre-residual queries/readouts.
    Per-side LayerNorm is shared across layers; flattened domain and cross-bus
    bundles are normalized separately. The host supplies the prediction head.
    """

    def __init__(self, num_layers: int, dim: int, num_user_tokens: int,
                 num_item_tokens: int, num_domains: int = 0):
        super().__init__()
        if min(num_layers, dim, num_user_tokens, num_item_tokens) < 1 or num_domains < 0:
            raise ValueError("dimensions must be positive; num_domains may be zero")
        self.user_shape = (num_layers, num_user_tokens, dim)
        self.item_shape = (num_layers, num_item_tokens, dim)
        self.domain_shape = (num_layers, num_domains, num_item_tokens)
        self.cross_width = num_layers * num_user_tokens * num_item_tokens
        self.domain_width = num_layers * num_domains * num_item_tokens
        if self.cross_width < 2 or self.domain_width == 1:
            raise ValueError("each normalized interaction bundle needs at least two values")
        self.user_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.item_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.cross_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.domain_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5) if self.domain_width else None

    @staticmethod
    def domain_interactions(queries: tf.Tensor, readouts: tf.Tensor,
                            valid_domains: tf.Tensor) -> tf.Tensor:
        """Raw aligned dots: [B,I,D], [B,S,I,D], bool [B,S] -> [B,S,I].

        Identity normalization matches the pinned author's block dots. True
        means available domain; mask before arithmetic. Final bundle LN may
        recenter zeros. This method performs no retrieval or attention.
        """
        tf.debugging.assert_type(valid_domains, tf.bool)
        tf.debugging.assert_shapes([(queries, ("B", "I", "D")),
                                    (readouts, ("B", "S", "I", "D")), (valid_domains, ("B", "S"))])
        safe = tf.where(valid_domains[..., None, None], readouts, 0)
        return tf.reduce_sum(queries[:, None] * safe, -1)

    def call(self, user_states: tf.Tensor, item_states: tf.Tensor,
             domain_dots: tf.Tensor | None = None) -> tf.Tensor:
        user_states = tf.ensure_shape(user_states, (None, *self.user_shape))
        item_states = tf.ensure_shape(item_states, (None, *self.item_shape))
        tf.debugging.assert_equal(tf.shape(user_states)[0], tf.shape(item_states)[0])
        batch = tf.shape(user_states)[0]
        cross = tf.linalg.matmul(self.user_norm(user_states), self.item_norm(item_states), transpose_b=True)
        cross = self.cross_norm(tf.reshape(cross, (batch, self.cross_width)))
        if self.domain_width:
            if domain_dots is None:
                raise ValueError("domain_dots are required when num_domains > 0")
            domain_dots = tf.ensure_shape(domain_dots, (None, *self.domain_shape))
            tf.debugging.assert_equal(tf.shape(domain_dots)[0], batch)
            domains = self.domain_norm(tf.reshape(domain_dots, (batch, self.domain_width)))
            return tf.concat((domains, cross), -1)
        if domain_dots is not None:
            raise ValueError("domain_dots require num_domains > 0")
        return cross
