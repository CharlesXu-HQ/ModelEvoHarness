"""Native sequence input contracts; True masks identify observed events."""
import tensorflow as tf


def validate_sequence_mask(values, valid=None, lengths=None, require_prefix=False, max_length=None):
    """Return [B,T] mask; lengths count valid events and alone imply prefixes.

    Explicit masks can have gaps unless require_prefix=True. max_length bounds
    padded time width. This checks layout, not feature meaning or provenance.
    """
    if not isinstance(require_prefix, bool):
        raise ValueError("require_prefix must be boolean")
    values = tf.convert_to_tensor(values)
    tf.debugging.assert_rank(values, 3)
    shape = tf.shape(values)
    tf.debugging.assert_positive(shape[0])
    tf.debugging.assert_positive(shape[2])
    if max_length is not None:
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        tf.debugging.assert_less_equal(shape[1], max_length, message="sequence exceeds max_length")
    if lengths is not None:
        lengths = tf.convert_to_tensor(lengths)
        if not lengths.dtype.is_integer:
            raise ValueError("lengths must be integer [B]")
        tf.debugging.assert_rank(lengths, 1)
        lengths = tf.cast(lengths, tf.int64)
        tf.debugging.assert_equal(tf.shape(lengths), shape[:1])
        tf.debugging.assert_greater_equal(lengths, tf.cast(0, lengths.dtype))
        tf.debugging.assert_less_equal(lengths, tf.cast(shape[1], lengths.dtype))
    if valid is None:
        valid = (tf.ones(shape[:2], tf.bool) if lengths is None else tf.sequence_mask(lengths, shape[1]))
    else:
        valid = tf.convert_to_tensor(valid)
        if valid.dtype != tf.bool:
            raise ValueError("valid must be a boolean [B,T] mask")
        tf.debugging.assert_rank(valid, 2)
        tf.debugging.assert_equal(tf.shape(valid), shape[:2])
    if lengths is not None:
        tf.debugging.assert_equal(tf.reduce_sum(tf.cast(valid, lengths.dtype), -1), lengths,
                                  message="lengths disagree with mask counts")
    if require_prefix:
        tf.debugging.assert_equal(tf.reduce_any(valid[:, 1:] & ~valid[:, :-1]), False,
                                  message="valid must be a right-padded prefix mask")
    return valid


def validate_aligned_sequences(attributes, masks, event_ids=None):
    """Check declared same-event attributes, masks and optional event identities.

    Masks only verify layout; the host must join attributes by event identity.
    Separate histories are separate groups. Padding event IDs are ignored.
    """
    if not isinstance(attributes, dict) or not attributes or set(attributes) != set(masks):
        raise ValueError("attributes and masks must have the same nonempty field keys")
    if event_ids is not None and set(event_ids) != set(attributes):
        raise ValueError("event_ids must identify every aligned attribute")
    reference_mask, reference_ids = None, None
    for name, value in attributes.items():
        value, mask = tf.convert_to_tensor(value), tf.convert_to_tensor(masks[name])
        tf.debugging.assert_rank_at_least(value, 2)
        if mask.dtype != tf.bool:
            raise ValueError("aligned attribute mask must be boolean")
        tf.debugging.assert_rank(mask, 2)
        tf.debugging.assert_equal(tf.shape(mask), tf.shape(value)[:2])
        if reference_mask is not None:
            tf.debugging.assert_equal(tf.shape(mask), tf.shape(reference_mask))
            tf.debugging.assert_equal(mask, reference_mask, message="aligned attribute masks must match")
        reference_mask = mask
        if event_ids is not None:
            ids = tf.convert_to_tensor(event_ids[name])
            tf.debugging.assert_rank(ids, 2)
            tf.debugging.assert_equal(tf.shape(ids), tf.shape(mask))
            if reference_ids is not None:
                tf.debugging.assert_equal(tf.boolean_mask(ids, mask), tf.boolean_mask(reference_ids, mask),
                                          message="aligned attributes have different valid event IDs")
            reference_ids = ids


class MaskedSequencePooling(tf.keras.layers.Layer):
    """Pool [B,T,D] to [B,D]; sanitize masked padding before arithmetic.

    Unnormalized weights are nonnegative: sum=sum(w*x), mean=sum(w*x)/sum(w),
    max=coordinatewise max(w*x). Softmax treats weights as logits over valid
    events. Empty sequences/zero-total-weight rows return zero. Valid vectors
    and weights must be finite; masked NaN/Inf padding is allowed.
    """
    def __init__(self, mode="mean", weight_normalization="none", require_prefix=False):
        super().__init__(autocast=False)
        if mode not in ("sum", "mean", "max"):
            raise ValueError("mode must be sum, mean or max")
        if weight_normalization not in ("none", "softmax"):
            raise ValueError("weight_normalization must be none or softmax")
        self.mode, self.weight_normalization, self.require_prefix = mode, weight_normalization, require_prefix

    def call(self, values, valid=None, lengths=None, weights=None):
        valid = validate_sequence_mask(values, valid, lengths, self.require_prefix)
        if not values.dtype.is_floating:
            raise ValueError("values must be real floating tensors")
        values = tf.where(valid[..., None], values, tf.zeros_like(values))
        tf.debugging.assert_all_finite(values, "valid sequence values must be finite")
        if weights is None:
            if self.weight_normalization == "softmax":
                raise ValueError("softmax normalization requires explicit weight logits")
            weights = tf.cast(valid, values.dtype)
        else:
            if not weights.dtype.is_floating:
                raise ValueError("weights must be floating [B,T]")
            tf.debugging.assert_rank(weights, 2)
            tf.debugging.assert_equal(tf.shape(weights), tf.shape(valid))
            weights = tf.where(valid, tf.cast(weights, values.dtype), tf.zeros_like(values[:, :, 0]))
            tf.debugging.assert_all_finite(weights, "valid sequence weights must be finite")
            if self.weight_normalization == "none":
                tf.debugging.assert_greater_equal(weights, tf.cast(0, weights.dtype))

        def pool():
            w = weights
            if self.weight_normalization == "softmax":
                logits = tf.where(valid, w, tf.cast(-float("inf"), w.dtype))
                logits = tf.where(tf.reduce_any(valid, 1, keepdims=True), logits, tf.zeros_like(logits))
                w = tf.where(valid, tf.nn.softmax(logits, axis=1), tf.zeros_like(w))
            if self.mode == "mean":
                scale = tf.reduce_max(w, 1, keepdims=True)
                w = w / tf.where(scale > 0, scale, tf.ones_like(scale))
                total = tf.reduce_sum(w, 1, keepdims=True)
                w = w / tf.where(total > 0, total, tf.ones_like(total))
            weighted = values * w[..., None]
            if self.mode != "max":
                return tf.reduce_sum(weighted, 1)
            result = tf.reduce_max(tf.where(valid[..., None], weighted,
                                           tf.cast(-float("inf"), values.dtype)), 1)
            return tf.where(tf.reduce_any(valid, 1, keepdims=True), result, tf.zeros_like(result))

        return tf.cond(tf.shape(values)[1] > 0, pool,
                       lambda: tf.reduce_sum(values, 1) + tf.reduce_sum(weights, 1, keepdims=True)*0)
