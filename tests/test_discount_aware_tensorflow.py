"""TensorFlow DANet-inspired branch algebra, masking and auxiliary supervision."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf


ERRORS = (ValueError, TypeError, tf.errors.InvalidArgumentError)


def _module():
    path = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models/tensorflow/discount_aware.py"
    assert path.is_file(), "Missing TensorFlow discount-aware component implementation"
    return importlib.import_module("model_evo_harness.models.tensorflow.discount_aware")


def _inputs(length=3):
    values = [tf.constant([[1., 0., 0., 0., 0.], [np.nan] * 5]),
              tf.constant([[True] * 5, [False] * 5]), tf.constant([[0.], [1.]]),
              tf.constant([[[0.], [np.log(3)], [np.nan]], [[np.nan]] * 3], tf.float32),
              tf.constant([[[2.], [4.], [np.nan]], [[np.nan]] * 3]),
              tf.constant([[True, True, False], [False] * 3]), tf.constant([[0.], [1.]])]
    if length == 0:
        values[3], values[4], values[5] = tf.zeros([2, 0, 1]), tf.zeros([2, 0, 1]), tf.zeros([2, 0], tf.bool)
    return values


def _dense(layer, kernel, bias):
    layer.kernel.assign(np.asarray(kernel, np.float32))
    layer.bias.assign(np.asarray(bias, np.float32))


def _dc_filter(model):
    first, last = np.zeros((6, 2)), np.zeros((2, 6))
    first[0, 0] = last[0, 0] = 1
    _dense(model.spectral_filter.layers[0], first, [0, 0])
    _dense(model.spectral_filter.layers[-1], last, np.zeros(6))


def test_odd_length_filter_and_attribute_pool_have_hand_computed_values():
    model = _module().DiscountPreferenceBranch(5, 1, 1, 1, 2)
    args = _inputs()
    model(*args)
    _dc_filter(model)
    _dense(model.attribute_attention.layers[0], [[1, 0], [0, 0]], [0, 0])
    _dense(model.attribute_attention.layers[-1], [[1], [0]], [0])
    _dense(model.user_projection, [[1, 2]], [0, 0])
    _dense(model.context_gate, [[0, 0]], [0, 0])
    result = model(*args)
    np.testing.assert_allclose(result["filtered_history"], [[.2] * 5, [0] * 5], atol=1e-6)
    np.testing.assert_allclose(result["residual_history"], [[.8, -.2, -.2, -.2, -.2], [0] * 5], atol=1e-6)
    np.testing.assert_allclose(result["attention"], [[.25, .75, 0], [0, 0, 0]], atol=1e-6)
    np.testing.assert_allclose(result["user_preference"], [[1.75, 3.5], [0, 0]], atol=1e-6)
    np.testing.assert_allclose(result["representation"],
                               [[1.75, 3.5, .8, -.2, -.2, -.2, -.2, .2, .2, .2, .2, .2], [0] * 12], atol=1e-6)


def test_target_changes_purchase_weights_and_promotion_changes_preference():
    model = _module().DiscountPreferenceBranch(5, 1, 1, 1, 2)
    args = _inputs()
    args[3] = tf.constant([[[-1.], [1.], [np.nan]], [[np.nan]] * 3])
    model(*args)
    _dense(model.attribute_attention.layers[0], [[1, 0], [-1, 0]], [0, 0])
    _dense(model.attribute_attention.layers[-1], [[1], [0]], [0])
    _dense(model.user_projection, [[1, 1]], [0, 0])
    _dense(model.context_gate, [[1, 1]], [0, 0])
    original = model(*args)
    args[2] = tf.constant([[2.], [1.]])
    changed_target = model(*args)
    np.testing.assert_allclose(changed_target["attention"][0], [.5, .5, 0], atol=1e-6)
    assert original["attention"][0, 1] > changed_target["attention"][0, 1]
    args[6] = tf.constant([[10.], [1.]])
    changed_context = model(*args)
    np.testing.assert_allclose(changed_context["attention"], changed_target["attention"], atol=1e-6)
    assert bool(tf.reduce_all(changed_context["user_preference"][0] > changed_target["user_preference"][0]))


@pytest.mark.parametrize("length", [0, 3])
def test_empty_or_masked_purchase_and_history_ignore_padding_and_learned_biases(length):
    model = _module().DiscountPreferenceBranch(5, 1, 1, 1, 2)
    args = _inputs(length)
    model(*args)
    for variable in model.trainable_variables:
        variable.assign(tf.ones_like(variable))
    result = model(*args)
    for value in result.values():
        assert np.isfinite(value.numpy()).all()
        np.testing.assert_array_equal(value[1], 0)
    assert result["attention"].shape == (2, length)
    if length == 0:
        np.testing.assert_array_equal(result["user_preference"], 0)
    clean = list(args)
    for index, mask_index in ((0, 1), (3, 5), (4, 5)):
        mask = args[mask_index] if index == 0 else args[mask_index][..., None]
        clean[index] = tf.where(mask, args[index], tf.zeros_like(args[index]))
    for name, value in model(*clean).items():
        np.testing.assert_array_equal(value, result[name])


def test_branch_gradients_flow_and_nan_padding_has_zero_gradient():
    model = _module().DiscountPreferenceBranch(5, 1, 1, 1, 2)
    args = _inputs()
    args[0] = tf.constant([[1., .3, np.nan, .2, .1], [np.nan] * 5])
    args[1] = tf.constant([[True, True, False, True, True], [False] * 5])
    variable_indices = (0, 2, 3, 4, 6)
    for index in variable_indices:
        args[index] = tf.Variable(args[index])
    model(*args)
    for variable in model.trainable_variables:
        variable.assign(tf.ones_like(variable) * .2)
    with tf.GradientTape() as tape:
        result = model(*args)
        loss = tf.reduce_sum(tf.square(result["representation"]))
    variables = [args[index] for index in variable_indices]
    gradients = tape.gradient(loss, variables + model.trainable_variables)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
    assert np.abs(gradients[0].numpy()[args[1].numpy()]).sum() > 0
    assert np.abs(gradients[3].numpy()[args[5].numpy()]).sum() > 0
    assert np.abs(gradients[4].numpy()).sum() > 0
    np.testing.assert_array_equal(gradients[0].numpy()[~args[1].numpy()], 0)
    np.testing.assert_array_equal(gradients[2].numpy()[~args[5].numpy()], 0)
    np.testing.assert_array_equal(gradients[3].numpy()[~args[5].numpy()], 0)


def test_heads_have_distinct_inputs_and_known_logits_and_discount_predictions():
    model = _module().DiscountAwareConversionHead(1, 1, 2)
    interest, discount = tf.constant([[2.], [4.]]), tf.constant([[1.], [2.]])
    model(interest, discount)
    _dense(model.conversion_head.layers[0], [[1, 0], [0, 1]], [0, 0])
    _dense(model.conversion_head.layers[-1], [[2], [3]], [1])
    _dense(model.discount_head.layers[0], [[1, 0]], [0, 0])
    _dense(model.discount_head.layers[-1], [[1], [0]], [0])
    outputs = model(interest, discount)
    np.testing.assert_allclose(outputs["conversion_logit"], [8., 15.], atol=1e-6)
    np.testing.assert_allclose(outputs["discount_prediction"], [0.73105858, 0.88079708], atol=1e-6)
    changed = model(interest + 10, discount)
    np.testing.assert_array_equal(changed["discount_prediction"], outputs["discount_prediction"])
    np.testing.assert_allclose(changed["conversion_logit"], outputs["conversion_logit"] + 20, atol=1e-6)


def test_objective_normalizes_only_observed_positive_discounts_and_detaches_labels():
    model = _module().DiscountAwareConversionHead(1, 1, 2)
    logits = tf.Variable([0., 0., 0., 0.])
    predictions = tf.Variable([.2, .3, .4, .5])
    conversion = tf.Variable([1., 0., 1., 1.])
    labels = tf.Variable([.8, np.nan, np.nan, .1])
    with tf.GradientTape(persistent=True) as tape:
        result = model.objective({"conversion_logit": logits, "discount_prediction": predictions},
                                 conversion, labels, tf.constant([True, True, False, True]), alpha=2.)
    np.testing.assert_allclose(result["conversion_loss"], np.log(2), atol=1e-6)
    np.testing.assert_allclose(result["discount_loss"], .26, atol=1e-6)
    np.testing.assert_allclose(result["loss"], np.log(2) + .52, atol=1e-6)
    assert int(result["discount_count"]) == 2
    np.testing.assert_allclose(tape.gradient(result["loss"], predictions), [-1.2, 0, 0, .8], atol=1e-6)
    np.testing.assert_allclose(tape.gradient(result["loss"], logits), [-.125, .125, -.125, -.125], atol=1e-6)
    assert tape.gradient(result["loss"], conversion) is None
    assert tape.gradient(result["loss"], labels) is None


@pytest.mark.parametrize("conversion", [[0., 0.], [1., 1.]])
def test_no_valid_positive_discount_is_differentiable_zero(conversion):
    model = _module().DiscountAwareConversionHead(1, 1, 2)
    predictions = tf.Variable([.2, .8])
    with tf.GradientTape() as tape:
        result = model.objective({"conversion_logit": tf.zeros([2]), "discount_prediction": predictions},
                                 tf.constant(conversion), tf.constant([np.nan, np.nan]), tf.constant([False, False]))
    assert float(result["discount_loss"]) == 0 and int(result["discount_count"]) == 0
    np.testing.assert_array_equal(tape.gradient(result["discount_loss"], predictions), [0, 0])
    assert np.isfinite(float(result["loss"]))


def test_tf_function_handles_dynamic_batch_and_zero_purchase_length():
    module = _module()
    branch, head = module.DiscountPreferenceBranch(5, 1, 1, 1, 2), module.DiscountAwareConversionHead(1, 12, 2)
    signatures = [tf.TensorSpec([None, 5], tf.float32), tf.TensorSpec([None, 5], tf.bool),
                  tf.TensorSpec([None, 1], tf.float32), tf.TensorSpec([None, None, 1], tf.float32),
                  tf.TensorSpec([None, None, 1], tf.float32), tf.TensorSpec([None, None], tf.bool),
                  tf.TensorSpec([None, 1], tf.float32)]

    @tf.function(input_signature=signatures)
    def run(*args):
        output = branch(*args)
        predictions = head(args[2], output["representation"])
        labels = tf.ones([tf.shape(args[2])[0]])
        loss = head.objective(predictions, labels, labels * .5, tf.cast(labels, tf.bool))
        return output, predictions, loss

    for length in (3, 0):
        output, predictions, losses = run(*_inputs(length))
        assert output["representation"].shape == (2, 12)
        assert predictions["conversion_logit"].shape == (2,)
        assert np.isfinite(float(losses["loss"]))
    output, _, _ = run(*(value[:1] for value in _inputs(0)))
    assert output["attention"].shape == (1, 0)


@pytest.mark.parametrize("which", ["history_width", "history_mask", "history_mask_dtype", "target_width", "purchase_length", "attributes_width", "context_batch", "context_rank", "observed_nan", "discount_range"])
def test_branch_rejects_invalid_shapes_masks_and_observed_values(which):
    model = _module().DiscountPreferenceBranch(5, 1, 1, 1, 2)
    args = _inputs()
    model(*args)
    if which == "history_width": args[0] = args[0][:, :4]
    elif which == "history_mask": args[1] = args[1][..., None]
    elif which == "history_mask_dtype": args[1] = tf.cast(args[1], tf.float32)
    elif which == "target_width": args[2] = tf.ones([2, 2])
    elif which == "purchase_length": args[3] = args[3][:, :2]
    elif which == "attributes_width": args[4] = tf.ones([2, 3, 2])
    elif which == "context_batch": args[6] = args[6][:1]
    elif which == "context_rank": args[6] = args[6][:, 0]
    elif which == "observed_nan": args[2] = tf.constant([[np.nan], [1.]])
    elif which == "discount_range": args[0] = tf.constant([[1.1, 0, 0, 0, 0], [np.nan] * 5])
    with pytest.raises(ERRORS): model(*args)


@pytest.mark.parametrize("which", ["conversion_shape", "nonbinary", "discount_shape", "valid_nan", "discount_range", "mask_dtype", "alpha_negative", "alpha_nan", "alpha_vector", "empty"])
def test_objective_rejects_invalid_supervision(which):
    model = _module().DiscountAwareConversionHead(1, 1, 2)
    outputs = {"conversion_logit": tf.zeros([2]), "discount_prediction": tf.constant([.2, .8])}
    conversion, labels, mask, alpha = tf.ones([2]), tf.constant([.3, .7]), tf.constant([True, True]), 1.
    if which == "conversion_shape": conversion = conversion[:, None]
    elif which == "nonbinary": conversion = tf.constant([1., .5])
    elif which == "discount_shape": labels = labels[:1]
    elif which == "valid_nan": labels = tf.constant([np.nan, .5])
    elif which == "discount_range": labels = tf.constant([1.1, .5])
    elif which == "mask_dtype": mask = tf.ones([2])
    elif which == "alpha_negative": alpha = -1.
    elif which == "alpha_nan": alpha = np.nan
    elif which == "alpha_vector": alpha = [1.]
    elif which == "empty":
        outputs = {name: value[:0] for name, value in outputs.items()}
        conversion, labels, mask = conversion[:0], labels[:0], mask[:0]
    with pytest.raises(ERRORS): model.objective(outputs, conversion, labels, mask, alpha=alpha)


def test_constructors_head_inputs_and_standalone_extraction():
    module = _module()
    for value in (0, -1, 2.5, True):
        with pytest.raises(ERRORS): module.DiscountPreferenceBranch(value, 1, 1, 1, 2)
        with pytest.raises(ERRORS): module.DiscountAwareConversionHead(value, 1, 2)
    head = module.DiscountAwareConversionHead(1, 1, 2)
    head(tf.ones([2, 1]), tf.ones([2, 1]))
    for interest, discount in ((tf.ones([2, 2]), tf.ones([2, 1])), (tf.ones([2, 1]), tf.ones([1, 1])),
                               (tf.ones([2, 1]), tf.constant([[np.nan], [1.]]))):
        with pytest.raises(ERRORS): head(interest, discount)
    namespace = {"__name__": "standalone_discount_aware"}
    exec(compile(Path(module.__file__).read_text(), "standalone_discount_aware.py", "exec"), namespace)
    result = namespace["DiscountPreferenceBranch"](5, 1, 1, 1, 2)(*_inputs())
    assert result["representation"].shape == (2, 12)


@pytest.mark.parametrize("which", ["complex_conversion", "complex_discount", "complex_alpha", "bool_alpha", "tensor_complex_alpha", "tensor_bool_alpha"])
def test_objective_rejects_lossy_supervision_types_before_cast(which):
    model = _module().DiscountAwareConversionHead(1, 1, 2)
    outputs = {"conversion_logit": tf.zeros([2]), "discount_prediction": tf.constant([.2, .8])}
    conversion, discount, alpha = tf.ones([2]), tf.constant([.3, .7]), 1.
    if which == "complex_conversion": conversion = tf.constant([1 + 2j, 1 + 0j])
    elif which == "complex_discount": discount = tf.constant([.3 + 2j, .7 + 0j])
    elif which == "complex_alpha": alpha = 1 + 2j
    elif which == "bool_alpha": alpha = True
    elif which == "tensor_complex_alpha": alpha = tf.constant(1 + 2j)
    elif which == "tensor_bool_alpha": alpha = tf.constant(True)
    with pytest.raises((TypeError, ValueError)):
        model.objective(outputs, conversion, discount, tf.constant([True, True]), alpha=alpha)


def test_objective_accepts_dynamic_real_scalar_tensor_alpha():
    model = _module().DiscountAwareConversionHead(1, 1, 2)

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def run(alpha):
        return model.objective({"conversion_logit": tf.zeros([2]), "discount_prediction": tf.constant([.2, .5])},
                               tf.constant([1., 0.]), tf.constant([.8, np.nan]), tf.constant([True, True]), alpha=alpha)

    for alpha in (0., 2.):
        result = run(tf.constant(alpha))
        np.testing.assert_allclose(result["loss"], np.log(2) + alpha * .36, atol=1e-6)
