"""TensorFlow objectives and host-controlled hard negative selection."""

import tensorflow as tf


def focal_loss(logits, labels, gamma=2.0, alpha=0.25):
    labels = tf.cast(labels, logits.dtype)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    p_t = tf.exp(-ce)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * ce)


def bpr_loss(positive_scores, negative_scores):
    return tf.reduce_mean(tf.nn.softplus(negative_scores - positive_scores))


def listwise_loss(scores, relevance, candidate_mask):
    masked_scores = tf.where(candidate_mask, scores,
                             tf.fill(tf.shape(scores), tf.cast(-1e9, scores.dtype)))
    masked_relevance = tf.cast(relevance, scores.dtype) * tf.cast(candidate_mask, scores.dtype)
    totals = tf.reduce_sum(masked_relevance, axis=-1, keepdims=True)
    assertion = tf.debugging.assert_positive(
        tf.reduce_sum(tf.cast(totals > 0, tf.int32)),
        message="listwise_loss needs at least one slate with observed relevance")
    with tf.control_dependencies([assertion] if assertion is not None else []):
        totals = tf.identity(totals)
    target = tf.math.divide_no_nan(masked_relevance, totals)
    per_slate = -tf.reduce_sum(target * tf.nn.log_softmax(masked_scores, axis=-1), axis=-1)
    return tf.reduce_mean(tf.boolean_mask(per_slate, tf.squeeze(totals > 0, axis=-1)))


def hard_negative_indices(queries, items, allowed_negative_mask, k):
    """The host supplies eligibility after excluding positives and unexposed items."""
    assertion = tf.debugging.assert_greater_equal(
        tf.reduce_min(tf.reduce_sum(tf.cast(allowed_negative_mask, tf.int32), axis=-1)), k,
        message="each query needs at least k eligible negative candidates")
    with tf.control_dependencies([assertion] if assertion is not None else []):
        allowed_negative_mask = tf.identity(allowed_negative_mask)
    scores = tf.linalg.matmul(queries, items, transpose_b=True)
    masked = tf.where(allowed_negative_mask, scores,
                      tf.fill(tf.shape(scores), tf.cast(-1e9, scores.dtype)))
    return tf.math.top_k(masked, k=k).indices
