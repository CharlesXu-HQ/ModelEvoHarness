"""Native representation recipes; host owns vocabulary identity and feature timing."""
import tensorflow as tf


class PretrainedIDFusion(tf.keras.layers.Layer):
    """Aligned pretrained table with init/sum/concat ID fusion and fixed padding.

    Freeze excludes the table from optimizer trainable variables, preserving
    lookup availability for pretrained-only and OOV rows.
    The supplied vocabulary identity and preprocessing remain host responsibilities.
    """
    def __init__(self, pretrained_vectors, output_dim, mode="init", freeze_pretrained=True, padding_idx=0):
        values = tf.convert_to_tensor(pretrained_vectors)
        if values.dtype.is_complex or values.shape.rank != 2 or any(d is None or d < 1 for d in values.shape):
            raise ValueError("pretrained_vectors must be a finite nonempty real matrix")
        if not values.dtype.is_floating:
            values = tf.cast(values, tf.float32)
        tf.debugging.assert_all_finite(values, "pretrained_vectors must be finite")
        super().__init__(dtype=values.dtype)
        if not isinstance(output_dim, int) or isinstance(output_dim, bool) or output_dim < 1:
            raise ValueError("output_dim must be positive")
        if mode not in ("init", "sum", "concat") or not isinstance(freeze_pretrained, bool):
            raise ValueError("invalid mode or freeze_pretrained")
        if padding_idx is not None and (not isinstance(padding_idx, int) or isinstance(padding_idx, bool) or
                                        not 0 <= padding_idx < values.shape[0]):
            raise ValueError("padding_idx must be a valid row or None")
        self.mode, self.padding_idx, self.vocab_size = mode, padding_idx, values.shape[0]
        self.pretrained = tf.keras.layers.Embedding(values.shape[0], values.shape[1],
            embeddings_initializer=tf.keras.initializers.Constant(values), trainable=not freeze_pretrained,
            dtype=values.dtype)
        self.pretrained.build((None,))
        self.id_embedding = (tf.keras.layers.Embedding(values.shape[0], output_dim, dtype=values.dtype)
                             if mode != "init" else None)
        if self.id_embedding is not None:
            self.id_embedding.build((None,))
        projection_dim = values.shape[1] + (output_dim if mode == "concat" else 0)
        self.projection = (tf.keras.layers.Dense(output_dim, use_bias=False, dtype=values.dtype)
                           if projection_dim != output_dim or mode == "concat" else tf.keras.layers.Identity(dtype=values.dtype))
        self.projection.build((None, projection_dim))

    def call(self, ids):
        if ids.dtype not in (tf.int32, tf.int64):
            raise ValueError("ids must be integer vocabulary rows")
        tf.debugging.assert_greater_equal(ids, tf.cast(0, ids.dtype))
        tf.debugging.assert_less(ids, tf.cast(self.vocab_size, ids.dtype))
        values = self.pretrained(ids)
        if self.mode == "concat":
            result = self.projection(tf.concat([values, self.id_embedding(ids)], -1))
        else:
            result = self.projection(values)
            if self.mode == "sum":
                result = result + self.id_embedding(ids)
        return (tf.where((ids != self.padding_idx)[..., None], result, tf.zeros_like(result))
                if self.padding_idx is not None else result)


class ResponseMLP(tf.keras.layers.Layer):
    """Dense features [B,D] to raw binary logit [B]; empty hidden_dims gives LR."""
    def __init__(self, input_dim, hidden_dims=()):
        super().__init__()
        widths = [input_dim, *hidden_dims]
        if any(not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in widths):
            raise ValueError("dimensions must be positive integers")
        self.input_dim = input_dim
        self.network = tf.keras.Sequential([tf.keras.layers.Dense(d, activation="relu")
                                            for d in hidden_dims] + [tf.keras.layers.Dense(1)])
        self.network.build((None, input_dim))

    def call(self, features):
        tf.debugging.assert_rank(features, 2)
        tf.debugging.assert_equal(tf.shape(features)[-1], self.input_dim)
        tf.debugging.assert_all_finite(features, "features must be finite")
        return tf.squeeze(self.network(features), -1)
