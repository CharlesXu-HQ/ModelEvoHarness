"""Standalone TensorFlow evidence for representation quality and feature screening.

Mechanism-inspired utilities, not a RankUp or TayFCS reproduction. Metrics do not
establish collapse, causal importance, or downstream improvement. Collect scores
on training data; removing an input still requires ablation and retraining.
"""

import math
import operator

import tensorflow as tf


def _divide(numerator, denominator):
    return tf.math.divide_no_nan(numerator, tf.cast(denominator, numerator.dtype))


def _masked(values, valid_mask):
    values = tf.convert_to_tensor(values)
    if not values.dtype.is_floating:
        raise ValueError("values need a real floating dtype")
    values = tf.stop_gradient(values)
    if values.dtype != tf.float64:
        values = tf.cast(values, tf.float32)
    checks = [tf.debugging.assert_rank(values, 3, message="values must have shape [B, T, D]"),
              tf.debugging.assert_positive(tf.shape(values)[0], message="batch must be nonempty"),
              tf.debugging.assert_positive(tf.shape(values)[2], message="channels must be nonempty")]
    if valid_mask is None:
        valid_mask = tf.ones(tf.shape(values)[:2], tf.bool)
    valid_mask = tf.convert_to_tensor(valid_mask)
    if valid_mask.dtype != tf.bool:
        raise ValueError("valid_mask must be boolean with shape [B, T]")
    checks.extend([tf.debugging.assert_rank(valid_mask, 2),
                   tf.debugging.assert_equal(tf.shape(valid_mask), tf.shape(values)[:2],
                                             message="valid_mask must have shape [B, T]")])
    with tf.control_dependencies([check for check in checks if check is not None]):
        values = tf.where(valid_mask[..., None], values, tf.zeros_like(values))
    check = tf.debugging.assert_all_finite(values, "valid values must be finite")
    with tf.control_dependencies([check] if check is not None else []):
        return tf.identity(values), tf.identity(valid_mask)


def _unit_scale(matrix):
    # The appended zero also defines the maximum for an empty token axis.
    flat = tf.reshape(tf.abs(matrix), (tf.shape(matrix)[0], -1))
    scale = tf.reduce_max(tf.concat((flat, tf.zeros((tf.shape(matrix)[0], 1), matrix.dtype)), 1), 1)
    return _divide(matrix, scale[:, None, None])


def _center(values, mask, count):
    # Shift first so mean rounding cannot invent rank for identical tokens.
    padded = tf.concat((values, tf.zeros((tf.shape(values)[0], 1, tf.shape(values)[2]), values.dtype)), 1)
    padded_mask = tf.concat((mask, tf.zeros((tf.shape(mask)[0], 1), tf.bool)), 1)
    first = tf.argmax(tf.cast(padded_mask, tf.int32), 1)
    origin = tf.gather(padded, first, batch_dims=1)
    shifted = tf.where(mask[..., None], values - origin[:, None, :], tf.zeros_like(values))
    mean = _divide(tf.reduce_sum(shifted, 1), count[:, None])
    return tf.where(mask[..., None], shifted - mean[:, None, :], tf.zeros_like(values))


class RepresentationDiagnostics:
    """Detached per-sample metrics for matrices with token rows and D columns.

    ``center=True`` subtracts each sample's valid-token channel mean before rank
    and branch similarity. The default uses uncentered matrices. Effective rank
    is exp(-sum(p*log(p))) with p=s/sum(s), using singular values (not squared
    values). Stable rank is sum(s**2)/max(s)**2. Both are zero for zero matrices.

    ``output_variance`` always means population variance across valid tokens,
    averaged over channels; this is not variance across the batch. Similarity
    is the signed Frobenius cosine of aligned, equally masked branch matrices.
    Zero-norm branches and samples with no valid tokens return zero. Half
    precision is promoted to float32; float64 inputs retain their precision.
    Tensor metrics work eagerly and inside tf.function.
    """

    def __init__(self, center=False):
        self.center = bool(center)

    def __call__(self, representations, valid_mask=None, reference=None):
        values, mask = _masked(representations, valid_mask)
        count = tf.reduce_sum(tf.cast(mask, tf.int64), 1)
        centered = _center(values, mask, count)
        variance = _divide(tf.reduce_sum(tf.square(centered), (1, 2)),
                           count * tf.cast(tf.shape(values)[2], tf.int64))
        matrix = _unit_scale(centered if self.center else values)
        # A zero row leaves the spectrum unchanged and supports T=0.
        padded = tf.concat((matrix, tf.zeros((tf.shape(matrix)[0], 1, tf.shape(matrix)[2]), matrix.dtype)), 1)
        singular = tf.linalg.svd(padded, compute_uv=False)
        probability = _divide(singular, tf.reduce_sum(singular, 1, keepdims=True))
        log_probability = tf.math.log(tf.where(probability > 0, probability, tf.ones_like(probability)))
        effective = tf.where(tf.reduce_sum(singular, 1) > 0,
                             tf.exp(-tf.reduce_sum(probability * log_probability, 1)),
                             tf.zeros_like(count, dtype=values.dtype))
        result = {
            "effective_rank": effective,
            "stable_rank": _divide(tf.reduce_sum(tf.square(singular), 1),
                                   tf.square(tf.reduce_max(singular, 1))),
            "output_variance": variance,
            "valid_tokens": count,
        }
        if reference is not None:
            check = tf.debugging.assert_equal(tf.shape(reference), tf.shape(representations),
                                              message="reference must have the same shape as representations")
            with tf.control_dependencies([check] if check is not None else []):
                other, _ = _masked(reference, mask)
            other = tf.cast(other, values.dtype)
            if self.center:
                other = _center(other, mask, count)
            other = _unit_scale(other)
            norm = tf.sqrt(tf.reduce_sum(tf.square(matrix), (1, 2)) * tf.reduce_sum(tf.square(other), (1, 2)))
            result["branch_similarity"] = tf.clip_by_value(
                _divide(tf.reduce_sum(matrix * other, (1, 2)), norm), -1, 1)
        return result

    def inspect(self, layers, valid_masks=None, references=None):
        """Inspect caller-captured named tensors; no model hooks are installed.

        Optional mappings provide masks or aligned reference branches by layer
        name. Missing entries use all tokens or omit branch similarity.
        """
        valid_masks, references = valid_masks or {}, references or {}
        return {name: self(value, valid_masks.get(name), references.get(name))
                for name, value in layers.items()}


class GradientFeatureSelector:
    """Training-data screening evidence for fields OR explicit cross tensors.

    Inputs and caller-computed objective gradients both have shape [B, F, D].
    Scores are mean_valid_B(abs(sum_D(activation * gradient))). This is a local
    first-order sensitivity, depends on objective reduction and feature scale,
    and cannot discover missing interactions from marginal scores.

    Redundancy is absolute uncentered cosine of each field pair flattened across
    their jointly valid batch rows and channels. ``overlap_counts`` makes absent
    evidence explicit; zero-norm/no-overlap pairs return zero. Redundancy is
    reported, never silently used to delete correlated fields. Call this method
    separately on explicit cross activations to score interactions themselves.
    Tensor metrics work eagerly and inside tf.function; selection is offline.
    """

    def __call__(self, activations, gradients, valid_mask=None):
        check = tf.debugging.assert_equal(tf.shape(activations), tf.shape(gradients),
                                          message="gradients must have the same shape as activations")
        with tf.control_dependencies([check] if check is not None else []):
            values, mask = _masked(activations, valid_mask)
        gradient, _ = _masked(gradients, mask)
        gradient = tf.cast(gradient, values.dtype)
        check = tf.debugging.assert_positive(tf.shape(values)[1], message="at least one field is required")
        with tf.control_dependencies([check] if check is not None else []):
            values = tf.identity(values)
        counts = tf.reduce_sum(tf.cast(mask, tf.int64), 0)
        saliency = _divide(tf.reduce_sum(tf.abs(tf.reduce_sum(values * gradient, 2)), 0), counts)
        # Rescale each field before dot products; cosine is scale invariant.
        fields = _unit_scale(tf.transpose(values, (1, 0, 2)))
        flat = tf.reshape(fields, (tf.shape(fields)[0], -1))
        pair_energy = tf.linalg.matmul(tf.reduce_sum(tf.square(fields), 2), tf.cast(mask, values.dtype))
        norm = tf.sqrt(pair_energy * tf.transpose(pair_energy))
        redundancy = tf.clip_by_value(_divide(tf.abs(tf.linalg.matmul(flat, flat, transpose_b=True)), norm), 0, 1)
        # int64 MatMul is CPU-only. Float64 preserves integer counts up to 2**53.
        count_mask = tf.cast(mask, tf.float64)
        return {"saliency": saliency, "redundancy": redundancy, "valid_counts": counts,
                "overlap_counts": tf.cast(tf.linalg.matmul(count_mask, count_mask, transpose_a=True), tf.int64)}

    def select(self, scores, budget, protected_indices=(), protected_groups=()):
        """Return an offline/eager proposal of original-order retained indices.

        Protected indices and ALL members of each protected interaction group
        are mandatory, even with zero marginal score. Overlapping groups form a
        union. Reject an insufficient budget; fill remaining slots by descending
        score, breaking ties by smaller index. No inputs are physically pruned.
        The caller must declare interaction-only fields/groups, then validate
        any proposed removal with ablation and retraining.
        """
        if not tf.executing_eagerly():
            raise ValueError("select is an eager/offline operation")
        scores = tf.convert_to_tensor(scores)
        if scores.shape.rank != 1:
            raise ValueError("scores must have shape [F]")
        scores = scores.numpy().tolist()
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("scores must be finite")
        if isinstance(budget, bool):
            raise ValueError("budget must be an integer between zero and field count")
        budget = operator.index(budget)
        if budget < 0 or budget > len(scores):
            raise ValueError("budget must be between zero and field count")
        protected = set()
        for group in (protected_indices, *protected_groups):
            for index in group:
                if isinstance(index, bool):
                    raise ValueError("protected indices must be integers")
                index = operator.index(index)
                if index < 0 or index >= len(scores):
                    raise ValueError("protected index is outside field range")
                protected.add(index)
        if len(protected) > budget:
            raise ValueError("budget cannot contain all protected fields and interaction groups")
        candidates = sorted((index for index in range(len(scores)) if index not in protected),
                            key=lambda index: (-scores[index], index))
        return tuple(sorted(protected.union(candidates[:budget - len(protected)])))
