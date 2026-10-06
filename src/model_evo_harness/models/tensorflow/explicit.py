"""Native TensorFlow building blocks for selected, mixed-field interactions.

Combine existing categorical embeddings with ``NumericFieldEmbedding`` outputs,
then use ``GroupedFM`` to expose separate within/between-group interaction terms.
Inputs must be preprocessed using training-split statistics. These blocks do not
infer semantic groups, bin numeric values, fuse logits, or add first-order terms.
"""

import tensorflow as tf


def _group_plan(field_count, groups, interactions):
    if type(field_count) is not int or field_count <= 0:
        raise ValueError("field_count must be a positive integer")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("groups must be a nonempty mapping")
    selected, used = {}, set()
    for name, indices in groups.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("group names must be nonempty strings")
        if not isinstance(indices, (list, tuple)) or not indices:
            raise ValueError("each group must contain field indices")
        for index in indices:
            if type(index) is not int or not 0 <= index < field_count:
                raise ValueError("field indices must be integers within field_count")
            if index in used:
                raise ValueError("groups must not repeat or overlap field indices")
            used.add(index)
        selected[name] = tuple(indices)
    if not isinstance(interactions, (list, tuple)) or not interactions:
        raise ValueError("at least one group interaction is required")
    pairs, seen = [], set()
    for pair in interactions:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each interaction must name two groups")
        left, right = pair
        if not all(isinstance(name, str) and name in selected for name in pair):
            raise ValueError("interaction groups must exist in groups")
        if left == right and len(selected[left]) < 2:
            raise ValueError("within-group interaction requires at least two fields")
        key = tuple(sorted(pair))
        if key in seen:
            raise ValueError("duplicate or reversed group interaction")
        seen.add(key)
        pairs.append((left, right))
    return selected, tuple(pairs)


class NumericFieldEmbedding(tf.keras.layers.Layer):
    """Encode each real-valued field as ``x_i * v_i``, without discretization.

    ``numeric`` is floating point ``[B,N]``; optional boolean ``present`` has the
    same shape. Missing values (including NaNs) become zero before multiplication.
    An observed zero also has a zero vector, but remains differentiable in x.
    Include separate missing-indicator fields if their effect should be learned.
    """

    def __init__(self, num_fields: int, embedding_dim: int):
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (num_fields, embedding_dim)):
            raise ValueError("num_fields and embedding_dim must be positive integers")
        self.num_fields = num_fields
        self.weight = self.add_weight(name="weight", shape=(num_fields, embedding_dim),
                                      initializer=tf.keras.initializers.RandomNormal(stddev=0.01))

    def call(self, numeric: tf.Tensor, present: tf.Tensor | None = None) -> tf.Tensor:
        tf.debugging.assert_rank(numeric, 2, message="numeric must have shape [batch, num_fields]")
        tf.debugging.assert_equal(tf.shape(numeric)[1], self.num_fields,
                                  message="numeric must have num_fields columns")
        if not numeric.dtype.is_floating:
            raise ValueError("numeric must be floating point")
        if present is not None:
            if present.dtype != tf.bool:
                raise ValueError("present must be a boolean mask matching numeric")
            tf.debugging.assert_rank(present, 2)
            tf.debugging.assert_equal(tf.shape(present), tf.shape(numeric),
                                      message="present must match numeric shape")
            numeric = tf.where(present, numeric, tf.zeros_like(numeric))
        return numeric[:, :, None] * self.weight[None, :, :]


class GroupedFM(tf.keras.layers.Layer):
    """Second-order terms for selected disjoint field groups; ``[B,F,D] -> [B,P]``.

    A pair ``(a,a)`` sums i<j within group a (no self-field products). ``(a,b)``
    sums all products across a and b. Output columns follow ``interactions``;
    a learned fusion layer or ablation mask may consume them separately.
    Groups may select a subset of input fields, but may not overlap. Computation
    takes O(B*(F+P)*D) without constructing the field-pair tensor.
    """

    def __init__(self, field_count: int, groups: dict[str, list[int]],
                 interactions: list[tuple[str, str]]):
        super().__init__()
        self.groups, self.interactions = _group_plan(field_count, groups, interactions)
        self.field_count = field_count

    def call(self, fields: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_rank(fields, 3, message="fields must have shape [batch, field_count, dim]")
        tf.debugging.assert_equal(tf.shape(fields)[1], self.field_count,
                                  message="fields must contain field_count fields")
        tf.debugging.assert_positive(tf.shape(fields)[2], message="embedding_dim must be positive")
        if not fields.dtype.is_floating:
            raise ValueError("fields must be floating point")
        selected = {name: tf.gather(fields, indices, axis=1) for name, indices in self.groups.items()}
        sums = {name: tf.reduce_sum(values, axis=1) for name, values in selected.items()}
        outputs = []
        for left, right in self.interactions:
            if left == right:
                term = 0.5 * (tf.square(sums[left]) - tf.reduce_sum(tf.square(selected[left]), axis=1))
            else:
                term = sums[left] * sums[right]
            outputs.append(tf.reduce_sum(term, axis=-1))
        return tf.stack(outputs, axis=1)
