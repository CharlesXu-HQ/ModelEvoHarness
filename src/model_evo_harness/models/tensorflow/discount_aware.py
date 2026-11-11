"""Original DANet-inspired discount branch and positive-only auxiliary objective.

Source: https://arxiv.org/html/2607.12578 . Native TensorFlow and stdlib only.
The host supplies the interest backbone, historical feature semantics, regular
calendar grid and pre-decision cutoff. Zero-filling missing history slots for
FFT is an explicit simplification; it does not assert zero observed discount.
Learned spectral filtering is not guaranteed low-pass. These components omit
the full production model/trainer and do not estimate causal coupon effects.
"""

from numbers import Integral

import tensorflow as tf


def _dimensions(*values):
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1 for value in values):
        raise ValueError("all constructor dimensions must be positive integers")


def _floating(value, name):
    value = tf.convert_to_tensor(value)
    if not value.dtype.is_floating:
        raise TypeError(f"{name} must have floating dtype")
    return value


def _matrix(value, width, name):
    value = _floating(value, name)
    tf.debugging.assert_rank(value, 2, message=f"{name} must have shape [B,{width}]")
    tf.debugging.assert_equal(tf.shape(value)[1], width, message=f"{name} width mismatch")
    tf.debugging.assert_all_finite(value, f"{name} observed values must be finite")
    return value


def _masked(value, mask, width, name):
    value, mask = _floating(value, name), tf.convert_to_tensor(mask)
    if mask.dtype != tf.bool:
        raise TypeError(f"{name} mask must be bool with True=valid")
    tf.debugging.assert_rank(value, 2 if width is None else 3, message=f"{name} rank mismatch")
    tf.debugging.assert_rank(mask, 2, message=f"{name} mask must have rank 2")
    tf.debugging.assert_equal(tf.shape(value)[:2], tf.shape(mask), message=f"{name} mask alignment mismatch")
    if width is not None:
        tf.debugging.assert_equal(tf.shape(value)[2], width, message=f"{name} width mismatch")
    clean = tf.where(mask if width is None else mask[..., None], value, tf.zeros_like(value))
    tf.debugging.assert_all_finite(clean, f"{name} observed values must be finite")
    return clean, mask


def _unit_interval(value, name):
    tf.debugging.assert_greater_equal(value, tf.cast(0, value.dtype), message=f"{name} must lie in [0,1]")
    tf.debugging.assert_less_equal(value, tf.cast(1, value.dtype), message=f"{name} must lie in [0,1]")


def _mlp(hidden_dim, output_dim, activation=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_dim, activation="relu"),
        tf.keras.layers.Dense(output_dim, activation=activation),
    ])


class DiscountPreferenceBranch(tf.keras.layers.Layer):
    """Return discount preference and fixed-length filtered/residual histories.

    Inputs: item_discount_history [B,N], bool history_mask [B,N], target_item
    [B,D], purchase_items [B,L,D], purchase_attributes [B,L,A], bool purchase_mask
    [B,L], promotion_context [B,C]. N is history_length; L may be zero. Observed
    history ratios use one host-defined [0,1] convention. Purchase attributes
    encode historical discount/category/price; no current outcome is an input.

    rFFT real/imaginary parts are concatenated, transformed by an MLP, and
    inverted with explicit length N. Attention scores concat(purchase,target)
    through a separate MLP and normalize over valid purchases. A projected
    attribute pool is multiplied by a sigmoid context gate. Softmax, sigmoid
    and the spectral MLP are engineering choices, not exact paper reproduction.

    Dict outputs: representation [B,H+2N] concatenates user_preference [B,H],
    residual_history [B,N], filtered_history [B,N]; attention is [B,L]. Masked
    slots and absent purchase representations are zero, including their input
    gradients. Valid observed inputs must be finite; padding may contain NaNs.
    """

    def __init__(self, history_length: int, item_dim: int, attribute_dim: int,
                 context_dim: int, hidden_dim: int):
        super().__init__()
        _dimensions(history_length, item_dim, attribute_dim, context_dim, hidden_dim)
        self.history_length = int(history_length)
        self.item_dim, self.attribute_dim, self.context_dim = int(item_dim), int(attribute_dim), int(context_dim)
        self.spectral_filter = _mlp(hidden_dim, 2 * (history_length // 2 + 1))
        self.attribute_attention = _mlp(hidden_dim, 1)
        self.user_projection = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.context_gate = tf.keras.layers.Dense(hidden_dim, activation="sigmoid")

    def call(self, item_discount_history, history_mask, target_item, purchase_items,
             purchase_attributes, purchase_mask, promotion_context):
        history, history_mask = _masked(item_discount_history, history_mask, None, "item_discount_history")
        tf.debugging.assert_equal(tf.shape(history)[1], self.history_length, message="history_length mismatch")
        _unit_interval(history, "observed historical discount ratios")
        target = _matrix(target_item, self.item_dim, "target_item")
        items, purchase_mask = _masked(purchase_items, purchase_mask, self.item_dim, "purchase_items")
        attributes, _ = _masked(purchase_attributes, purchase_mask, self.attribute_dim, "purchase_attributes")
        context = _matrix(promotion_context, self.context_dim, "promotion_context")
        for value in (target, items, context):
            tf.debugging.assert_equal(tf.shape(value)[0], tf.shape(history)[0], message="input batch mismatch")

        # FFT kernels require float32/float64; use float32 for lower precision inputs.
        work = tf.cast(history, tf.float32) if history.dtype in (tf.float16, tf.bfloat16) else history
        spectrum = tf.signal.rfft(work, fft_length=[self.history_length])
        features = tf.concat([tf.math.real(spectrum), tf.math.imag(spectrum)], axis=-1)
        spectral_output = tf.cast(self.spectral_filter(features), work.dtype)
        real, imaginary = tf.split(spectral_output, 2, axis=-1)
        filtered = tf.cast(tf.signal.irfft(tf.complex(real, imaginary), fft_length=[self.history_length]), history.dtype)
        filtered = tf.where(history_mask, filtered, tf.zeros_like(filtered))
        residual = history - filtered

        target_tokens = tf.broadcast_to(target[:, None, :], tf.shape(items))
        scores = tf.squeeze(self.attribute_attention(tf.concat([items, target_tokens], axis=-1)), axis=-1)
        scores = tf.cast(scores, tf.float32) if scores.dtype in (tf.float16, tf.bfloat16) else scores
        present = tf.reduce_any(purchase_mask, axis=1, keepdims=True)
        # Only rows with no purchases can use the zero sentinel.
        scores = tf.concat([scores, tf.zeros([tf.shape(scores)[0], 1], scores.dtype)], axis=1)
        available = tf.concat([purchase_mask, ~present], axis=1)
        scores = tf.where(available, scores, tf.cast(float("-inf"), scores.dtype))
        attention = tf.nn.softmax(scores, axis=1)[:, :-1]
        pooled = tf.reduce_sum(attention[..., None] * tf.cast(attributes, attention.dtype), axis=1)
        preference = self.user_projection(pooled) * self.context_gate(context)
        preference = tf.where(present, preference, tf.zeros_like(preference))
        representation = tf.concat([preference, residual, filtered], axis=-1)
        return {"representation": representation, "user_preference": preference,
                "filtered_history": filtered, "residual_history": residual, "attention": attention}


class DiscountAwareConversionHead(tf.keras.layers.Layer):
    """Conversion from interest+discount, and discount regression from discount only.

    call takes interest_representation [B,I], discount_representation [B,P] and
    returns conversion_logit [B] plus sigmoid discount_prediction [B]. The host
    chooses the interest backbone. Label supervision is confined to objective;
    no discount label or purchase outcome participates in forward prediction.
    """

    def __init__(self, interest_dim: int, discount_dim: int, hidden_dim: int):
        super().__init__()
        _dimensions(interest_dim, discount_dim, hidden_dim)
        self.interest_dim, self.discount_dim = int(interest_dim), int(discount_dim)
        self.conversion_head = _mlp(hidden_dim, 1)
        self.discount_head = _mlp(hidden_dim, 1, activation="sigmoid")

    def call(self, interest_representation, discount_representation):
        interest = _matrix(interest_representation, self.interest_dim, "interest_representation")
        discount = _matrix(discount_representation, self.discount_dim, "discount_representation")
        tf.debugging.assert_equal(tf.shape(interest)[0], tf.shape(discount)[0], message="input batch mismatch")
        return {"conversion_logit": tf.squeeze(self.conversion_head(tf.concat([interest, discount], axis=-1)), -1),
                "discount_prediction": tf.squeeze(self.discount_head(discount), -1)}

    def objective(self, outputs, conversion_labels, discount_labels, discount_label_mask, alpha=1.0):
        """Mean conversion BCE + alpha * valid-purchase-only mean discount MSE.

        All labels/mask have shape [B]; B must be nonzero. Conversion labels are
        binary and observed positive discount labels lie in [0,1]. Unavailable
        or negative-row discount labels are ignored before arithmetic, including
        NaNs. No-valid-positive batches give differentiable zero auxiliary loss.
        Labels must be real and are detached; alpha is a finite nonnegative real
        scalar, excluding bool. Floating scalar tensors are supported. Returns scalar
        loss, conversion_loss, discount_loss and integer discount_count.
        """
        logits = _floating(outputs["conversion_logit"], "conversion_logit")
        prediction = _floating(outputs["discount_prediction"], "discount_prediction")
        tf.debugging.assert_rank(logits, 1, message="conversion_logit must have shape [B]")
        tf.debugging.assert_positive(tf.shape(logits)[0], message="objective requires a nonempty batch")
        tf.debugging.assert_rank(prediction, 1, message="discount_prediction must have shape [B]")
        tf.debugging.assert_equal(tf.shape(prediction), tf.shape(logits), message="output shape mismatch")
        tf.debugging.assert_all_finite(logits, "conversion logits must be finite")
        tf.debugging.assert_all_finite(prediction, "discount predictions must be finite")
        _unit_interval(prediction, "discount predictions")
        dtype = tf.float32 if logits.dtype in (tf.float16, tf.bfloat16) else logits.dtype
        logits, prediction = tf.cast(logits, dtype), tf.cast(prediction, dtype)
        conversion = tf.convert_to_tensor(conversion_labels)
        labels = tf.convert_to_tensor(discount_labels)
        if conversion.dtype.is_complex or labels.dtype.is_complex:
            raise TypeError("conversion and discount labels must be real")
        conversion = tf.stop_gradient(tf.cast(conversion, dtype))
        labels = tf.stop_gradient(tf.cast(labels, dtype))
        mask = tf.convert_to_tensor(discount_label_mask)
        if mask.dtype != tf.bool:
            raise TypeError("discount_label_mask must be bool with True=valid")
        for value in (conversion, labels, mask):
            tf.debugging.assert_rank(value, 1, message="labels and mask must have shape [B]")
            tf.debugging.assert_equal(tf.shape(value), tf.shape(logits), message="label/output shape mismatch")
        tf.debugging.assert_all_finite(conversion, "conversion labels must be finite")
        tf.debugging.assert_equal(tf.reduce_all((conversion == 0) | (conversion == 1)), True,
                                  message="conversion labels must be binary")
        valid = (conversion == 1) & mask
        labels = tf.where(valid, labels, tf.zeros_like(labels))
        tf.debugging.assert_all_finite(labels, "observed positive discount labels must be finite")
        _unit_interval(labels, "observed positive discount labels")
        weight = tf.convert_to_tensor(alpha)
        if not (weight.dtype.is_floating or weight.dtype.is_integer):
            raise TypeError("alpha must be a real scalar, excluding bool")
        weight = tf.cast(weight, dtype)
        tf.debugging.assert_rank(weight, 0, message="alpha must be scalar")
        tf.debugging.assert_all_finite(weight, "alpha must be finite")
        tf.debugging.assert_greater_equal(weight, tf.cast(0, dtype), message="alpha must be nonnegative")
        conversion_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=conversion, logits=logits))
        error = tf.where(valid, prediction, tf.zeros_like(prediction)) - labels
        count = tf.reduce_sum(tf.cast(valid, tf.int32))
        discount_loss = tf.reduce_sum(tf.square(error)) / tf.cast(tf.maximum(count, 1), dtype)
        return {"loss": conversion_loss + weight * discount_loss, "conversion_loss": conversion_loss,
                "discount_loss": discount_loss, "discount_count": count}
