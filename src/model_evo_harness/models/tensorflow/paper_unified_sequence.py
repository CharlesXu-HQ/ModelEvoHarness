"""Content-only UniDot conditional gates and multi-channel pooling (Keras).

Adapted from linrongc/kdd26_unidot at
1399702006a1fc2f86329c8c6133f3f3ac1e6d0d: common.py CondGatedSwiGLU and
unified_model.py MultiChannelSeqPool. Changes: TensorFlow port, True=valid
masks, sanitized padding, zero empty domains after affine transforms, stable
half-type L2 with zero-direction gradients, internal condition broadcast,
no temporal/interaction variants or data dependencies.

MIT License
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


class ConditionalSwiGLU(tf.keras.Model):
    """History [B,L,D_s], condition [B,D_c], bool valid mask [B,L] -> [B,L,D_o].

    out = Wo(value(history) * SiLU(gate([history, condition]))).
    The condition never enters the value projection. Padding returns zero.
    """

    def __init__(self, sequence_dim: int, condition_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        if min(sequence_dim, condition_dim, hidden_dim, output_dim) < 1:
            raise ValueError("dimensions must be positive")
        self.sequence_dim, self.condition_dim = sequence_dim, condition_dim
        self.value = tf.keras.layers.Dense(hidden_dim)
        self.gate = tf.keras.layers.Dense(hidden_dim)
        self.output_projection = tf.keras.layers.Dense(output_dim)

    def call(self, history: tf.Tensor, condition: tf.Tensor,
             valid_mask: tf.Tensor) -> tf.Tensor:
        history = tf.ensure_shape(history, (None, None, self.sequence_dim))
        condition = tf.ensure_shape(condition, (None, self.condition_dim))
        tf.debugging.assert_type(valid_mask, tf.bool)
        tf.debugging.assert_shapes([(history, ("B", "L", self.sequence_dim)),
                                    (condition, ("B", self.condition_dim)), (valid_mask, ("B", "L"))])
        safe = tf.where(valid_mask[..., None], history, 0)
        context = tf.broadcast_to(condition[:, None], (tf.shape(history)[0], tf.shape(history)[1], self.condition_dim))
        gate = tf.nn.silu(self.gate(tf.concat((safe, context), -1)))
        output = self.output_projection(self.value(safe) * gate)
        return tf.where(valid_mask[..., None], output, 0)


class MultiChannelSequencePool(tf.keras.Model):
    """State [B,T,D], S histories [B,L_s,D], S bool valid masks -> [B,S*C,D].

    Independent sigmoid channels pool each domain, followed by L2 normalization,
    domain-specific projection and shared LayerNorm. Empty domains (including
    L=0) return zero after all affine operations. State tokens are fully defined.
    """

    def __init__(self, dim: int, num_state_tokens: int, num_channels: int, num_domains: int):
        super().__init__()
        if min(dim, num_state_tokens, num_channels, num_domains) < 1:
            raise ValueError("dimensions and domain count must be positive")
        self.dim, self.num_state_tokens = dim, num_state_tokens
        self.num_domains, self.num_channels = num_domains, num_channels
        self.state_projection = tf.keras.layers.Dense(dim)
        self.scorers = [tf.keras.Sequential((tf.keras.layers.Dense(dim, activation=tf.nn.gelu),
                                             tf.keras.layers.Dense(num_channels))) for _ in range(num_domains)]
        self.pool_projections = [tf.keras.layers.Dense(dim) for _ in range(num_domains)]
        self.out_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, state_tokens: tf.Tensor, histories: list[tf.Tensor],
             valid_masks: list[tf.Tensor]) -> tf.Tensor:
        if len(histories) != self.num_domains or len(valid_masks) != self.num_domains:
            raise ValueError("history/mask lists must match the configured domain count")
        state_tokens = tf.ensure_shape(state_tokens, (None, self.num_state_tokens, self.dim))
        batch = tf.shape(state_tokens)[0]
        state = self.state_projection(tf.reshape(state_tokens, (batch, self.num_state_tokens * self.dim)))
        result = []
        for index, (history, mask) in enumerate(zip(histories, valid_masks)):
            history = tf.ensure_shape(history, (None, None, self.dim))
            tf.debugging.assert_type(mask, tf.bool)
            tf.debugging.assert_shapes([(state_tokens, ("B", self.num_state_tokens, self.dim)),
                                        (history, ("B", "L", self.dim)), (mask, ("B", "L"))])
            safe = tf.where(mask[..., None], history, 0)
            context = tf.broadcast_to(state[:, None], tf.shape(history))
            weights = tf.sigmoid(self.scorers[index](tf.concat((safe, context), -1)))
            weights = tf.where(mask[..., None], weights, 0)
            pooled = tf.einsum("blc,bld->bcd", weights, safe)
            # TF clamps squared norm; float32 keeps the 1e-12 norm floor for half types.
            norm_input = tf.cast(pooled, tf.float32) if pooled.dtype in (tf.float16, tf.bfloat16) else pooled
            normalized = tf.math.l2_normalize(norm_input, axis=-1, epsilon=1e-24)
            # Zero vectors have no direction; define their normalization gradient as zero.
            normalized = tf.where(tf.reduce_any(tf.not_equal(norm_input, 0), -1, keepdims=True), normalized, 0)
            pooled = tf.cast(normalized, pooled.dtype)
            output = self.out_norm(self.pool_projections[index](pooled))
            result.append(tf.where(tf.reduce_any(mask, -1)[:, None, None], output, 0))
        return tf.concat(result, 1)
