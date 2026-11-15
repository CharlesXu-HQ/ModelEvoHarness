"""Host-controlled conditional negative sampling with stateless randomness."""
import math
from numbers import Real

import tensorflow as tf


class ConditionalNegativeSampler(tf.keras.layers.Layer):
    """With-replacement sampling over host-supplied counts[N] and bool masks[B,N].

    Uniform uses base = 1/N. Popularity uses p = counts**alpha/sum(counts**alpha),
    then base = (1-uniform_mix)*p + uniform_mix/N. Condition base on
    eligible_mask & ~positive_mask for every row. uniform_mix is a catalog
    mixture weight, not the final per-row uniform share. Alpha zero gives
    equal weights; a used popularity branch still requires positive count mass.

    Return item_ids[B,K] and sampling_probs[B,K], each draw's conditional q.
    Sampling is with replacement, not an inclusion-probability calculation.
    Hosts own training-period counts, historical eligibility, known positives,
    and false-negative audits. Unobserved samples are not verified dislikes;
    q is not exposure propensity. This is not an implementation of RALM.
    Materializes O(B*N) probabilities; hosts must bound the candidate pool.
    Seeded stateless draws preserve global RNG state; vary seed for fresh draws.
    Seed is a Python integer in [0, 2**63-1] in both frameworks.
    Probabilities are normalized in float64 log space before the output cast.
    """

    def __init__(self, distribution='uniform', alpha=.75, uniform_mix=0.):
        super().__init__(autocast=False)
        if distribution not in ('uniform', 'popularity'):
            raise ValueError('distribution must be uniform or popularity')
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')
        if (isinstance(uniform_mix, bool) or not isinstance(uniform_mix, Real)
                or not math.isfinite(uniform_mix) or not 0 <= uniform_mix <= 1):
            raise ValueError('uniform_mix must be finite and in [0,1]')
        self.distribution, self.alpha, self.uniform_mix = distribution, alpha, uniform_mix

    def call(self, counts, eligible_mask, positive_mask, num_samples, seed=0):
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError('num_samples must be a positive integer')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63-1]')
        if eligible_mask.dtype != tf.bool or positive_mask.dtype != tf.bool:
            raise ValueError('masks must be boolean')
        if counts.dtype.is_complex or counts.dtype == tf.bool:
            raise ValueError('counts must be real numeric values, not complex or boolean')
        output_dtype = tf.float64 if counts.dtype == tf.float64 else tf.float32
        counts = tf.stop_gradient(tf.cast(counts, tf.float64))
        checks = [tf.debugging.assert_rank(counts, 1),
                  tf.debugging.assert_rank(eligible_mask, 2),
                  tf.debugging.assert_rank(positive_mask, 2),
                  tf.debugging.assert_positive(tf.size(counts)),
                  tf.debugging.assert_equal(tf.shape(eligible_mask), tf.shape(positive_mask)),
                  tf.debugging.assert_equal(tf.shape(eligible_mask)[1], tf.size(counts)),
                  tf.debugging.assert_all_finite(counts, 'counts must be finite'),
                  tf.debugging.assert_non_negative(counts)]
        with tf.control_dependencies([check for check in checks if check is not None]):
            counts = tf.identity(counts)
            allowed = tf.identity(eligible_mask) & ~positive_mask
        log_uniform = tf.fill(tf.shape(counts), -tf.math.log(tf.cast(tf.size(counts), counts.dtype)))
        log_base, scale = log_uniform, 1.
        if self.distribution == 'popularity' and self.uniform_mix < 1:
            check = tf.debugging.assert_positive(tf.reduce_max(counts),
                                                message='popularity requires positive catalog count mass')
            with tf.control_dependencies([check] if check is not None else []):
                log_counts = tf.math.log(counts) if self.alpha > 0 else tf.zeros_like(counts)
            if self.uniform_mix == 0:
                # Cancel the catalog normalizer and center per row before alpha
                # scaling, preserving tiny eligible support even for huge alpha.
                log_base, scale = log_counts, self.alpha if self.alpha > 0 else 1.
            else:
                logits = ((log_counts - tf.reduce_max(log_counts)) * self.alpha
                          if self.alpha > 0 else log_counts)
                log_popularity = logits - tf.reduce_logsumexp(logits)
                log_base = tf.reduce_logsumexp(tf.stack([
                    log_popularity + math.log1p(-self.uniform_mix),
                    log_uniform + math.log(self.uniform_mix)]), axis=0)
        logits = tf.where(allowed, log_base[None, :], tf.constant(-math.inf, counts.dtype))
        maximum = tf.reduce_max(logits, axis=-1, keepdims=True)
        check = tf.debugging.assert_all_finite(maximum,
                                               'each query needs positive eligible non-positive probability mass')
        with tf.control_dependencies([check] if check is not None else []):
            logits = (logits - maximum) * scale
        log_probabilities = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
        item_ids = tf.random.stateless_categorical(log_probabilities, num_samples,
                                                  seed=tf.constant([seed, 0], tf.int64))
        return {'item_ids': item_ids,
                'sampling_probs': tf.cast(tf.gather(tf.exp(log_probabilities), item_ids, batch_dims=1), output_dtype)}

    def sample(self, counts, eligible_mask, positive_mask, num_samples, seed=0):
        return self(counts, eligible_mask, positive_mask, num_samples=num_samples, seed=seed)
