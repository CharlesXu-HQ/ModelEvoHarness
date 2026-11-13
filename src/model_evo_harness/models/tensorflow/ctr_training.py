"""Independent CTR auxiliary objectives. Host supplies aligned tensors and groups."""
import math

import tensorflow as tf


def _weight(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("weights must be finite nonnegative numbers")
    return float(value)


def _finite(value):
    value = tf.convert_to_tensor(value)
    if not value.dtype.is_floating:
        raise ValueError("inputs must be real floating tensors")
    tf.debugging.assert_all_finite(value, "inputs must be finite")
    return value


def _labels(labels, batch, dtype):
    labels = tf.convert_to_tensor(labels)
    if labels.dtype.is_complex:
        raise ValueError("labels must be real binary values")
    tf.debugging.assert_rank(labels, 1)
    tf.debugging.assert_equal(tf.shape(labels)[0], batch)
    tf.debugging.assert_equal(tf.reduce_all((labels == 0) | (labels == 1)), True, message="labels must be binary")
    return tf.stop_gradient(tf.cast(labels, dtype))


def _mean(x, axis=None):
    # Divide before summing: finite large entries can overflow a sum-first mean.
    count = tf.size(x) if axis is None else tf.shape(x)[axis]
    return tf.reduce_sum(x / tf.cast(count, x.dtype), axis)


class EnsembleConsistency:
    """BCE plus sum of branch BCE to a detached ensemble, logits [B,M].

    logit_mean follows FinalNet; probability_mean averages predictions. The
    explicit weight is a local adaptation. No observed auxiliary labels are added.
    """
    def __init__(self, weight=1.0, aggregation="logit_mean"):
        self.weight = _weight(weight)
        if aggregation not in ("logit_mean", "probability_mean"):
            raise ValueError("aggregation must be logit_mean or probability_mean")
        self.aggregation = aggregation

    def __call__(self, logits, labels):
        logits = _finite(logits)
        tf.debugging.assert_rank(logits, 2)
        tf.debugging.assert_positive(tf.shape(logits))
        labels = _labels(labels, tf.shape(logits)[0], logits.dtype)
        if self.aggregation == "logit_mean":
            mean = _mean(logits, -1)
            supervised = _mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=mean))
            teacher = tf.stop_gradient(tf.sigmoid(mean))
        else:
            log_count = tf.math.log(tf.cast(tf.shape(logits)[1], logits.dtype))
            log_p = tf.reduce_logsumexp(tf.math.log_sigmoid(logits), -1) - log_count
            log_q = tf.reduce_logsumexp(tf.math.log_sigmoid(-logits), -1) - log_count
            supervised = -_mean(labels*log_p + (1-labels)*log_q)
            teacher = tf.stop_gradient(tf.exp(log_p))
        branch_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.broadcast_to(teacher[:, None], tf.shape(logits)), logits=logits)
        branch_losses = _mean(branch_loss, 0)
        consistency = tf.reduce_sum(branch_losses)
        return {"loss": supervised + tf.reduce_sum(self.weight*branch_losses), "supervised": supervised,
                "consistency": consistency, "teacher": teacher}


class NextBehaviorObjective(tf.keras.layers.Layer):
    """Aligned state/positive/negative transitions [B,T,*] and bool mask [B,T].

    Host shifts events and samples negatives. Mean positive/negative BCE over
    valid transitions; empty transition sets contribute differentiable zero.
    """
    def __init__(self, state_dim, event_dim, hidden_dim):
        super().__init__()
        if any(not isinstance(d, int) or isinstance(d, bool) or d < 1
               for d in (state_dim, event_dim, hidden_dim)):
            raise ValueError("dimensions must be positive integers")
        self.state_dim, self.event_dim = state_dim, event_dim
        self.scorer = tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim, activation="relu"),
                                           tf.keras.layers.Dense(1)])
        self.scorer.build((None, None, state_dim + event_dim))

    def call(self, states, positive, negative, valid):
        if valid.dtype != tf.bool:
            raise ValueError("transition mask must be boolean")
        tf.debugging.assert_rank(states, 3)
        tf.debugging.assert_positive(tf.shape(states)[0])
        tf.debugging.assert_equal(tf.shape(states)[-1], self.state_dim)
        tf.debugging.assert_equal(tf.shape(positive), tf.concat([tf.shape(states)[:2], [self.event_dim]], 0))
        tf.debugging.assert_equal(tf.shape(negative), tf.shape(positive))
        tf.debugging.assert_equal(tf.shape(valid), tf.shape(states)[:2])
        states, positive, negative = [_finite(tf.where(valid[..., None], x, tf.zeros_like(x)))
                                      for x in (states, positive, negative)]
        pos = tf.squeeze(self.scorer(tf.concat([states, positive], -1)), -1)
        neg = tf.squeeze(self.scorer(tf.concat([states, negative], -1)), -1)
        count = tf.reduce_sum(tf.cast(valid, tf.int32))
        denominator = tf.cast(2*tf.maximum(count, 1), pos.dtype)
        terms = tf.nn.softplus(-pos) / denominator + tf.nn.softplus(neg) / denominator
        loss = tf.reduce_sum(tf.where(valid, terms, tf.zeros_like(pos)))
        return {"loss": loss, "positive_logit": tf.where(valid, pos, tf.zeros_like(pos)),
                "negative_logit": tf.where(valid, neg, tf.zeros_like(neg)), "valid_count": count}


class CoarseFineObjective:
    """Weighted coarse/fine BCE on the same observed labels, raw scores [B]."""
    def __init__(self, coarse_weight=1.0, fine_weight=1.0):
        self.coarse_weight, self.fine_weight = _weight(coarse_weight), _weight(fine_weight)

    def __call__(self, coarse_logit, fine_logit, labels):
        coarse_logit, fine_logit = _finite(coarse_logit), _finite(fine_logit)
        tf.debugging.assert_rank(coarse_logit, 1)
        tf.debugging.assert_positive(tf.shape(coarse_logit)[0])
        tf.debugging.assert_equal(tf.shape(fine_logit), tf.shape(coarse_logit))
        labels = _labels(labels, tf.shape(coarse_logit)[0], coarse_logit.dtype)
        coarse = _mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=coarse_logit))
        fine = _mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=fine_logit))
        return {"loss": self.coarse_weight*coarse + self.fine_weight*fine, "coarse": coarse, "fine": fine}


class GroupedRegularization:
    """Explicit lambda/p * sum(abs(parameter)**p); host declares trainable groups.

    No implicit parameter-name selection or decoupled optimizer weight decay.
    The plain callable preserves variable identity when checking group overlap.
    """
    def __init__(self, embedding_weight=0.0, network_weight=0.0, p=2):
        self.embedding_weight, self.network_weight = _weight(embedding_weight), _weight(network_weight)
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 1:
            raise ValueError("p must be finite and >= 1")
        self.p = p

    def __call__(self, embedding_parameters, network_parameters):
        groups = [list(embedding_parameters), list(network_parameters)]
        parameters = groups[0] + groups[1]
        if not parameters or len({id(x) for x in parameters}) != len(parameters):
            raise ValueError("parameter groups must be nonempty in total with no overlap or duplicate")
        zero = tf.reduce_sum(_finite(parameters[0])*0)
        penalties = [sum((tf.reduce_sum(_finite(x)*0) for x in group), zero) if weight == 0 else
                     weight/self.p * sum((tf.reduce_sum(tf.abs(_finite(x))**self.p)
                                         for x in group), zero)
                     for group, weight in zip(groups, (self.embedding_weight, self.network_weight))]
        return {"loss": sum(penalties), "embedding": penalties[0], "network": penalties[1]}
