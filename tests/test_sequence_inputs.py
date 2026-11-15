"""Explicit sequence contracts and hand-computed pooling/training scales."""
import importlib
import math
import os

import numpy as np
import pytest


REQUIRE_GPU = os.environ.get("MODELEVO_REQUIRE_GPU") == "1"


def runtime(framework):
    name = "torch" if framework == "pytorch" else "tensorflow"
    try:
        lib = importlib.import_module(name)
    except ModuleNotFoundError:
        if REQUIRE_GPU:
            pytest.fail(f"{name} is required for GPU validation")
        pytest.skip(f"{name} is unavailable")
    if REQUIRE_GPU:
        available = lib.cuda.is_available() if framework == "pytorch" else bool(lib.config.list_physical_devices("GPU"))
        assert available, f"{framework} GPU is required"
    return lib


@pytest.fixture(params=["pytorch", "tensorflow"])
def native(request):
    framework = request.param
    lib = runtime(framework)
    module = importlib.import_module(f"model_evo_harness.models.{framework}.sequence_inputs")
    if framework == "pytorch":
        device = "cuda:0" if REQUIRE_GPU else "cpu"
        yield module, lambda x: lib.tensor(x, dtype=lib.float32, device=device), lambda x: lib.tensor(x, device=device), lib
    else:
        with lib.device("/GPU:0" if REQUIRE_GPU else "/CPU:0"):
            yield module, lambda x: lib.constant(x, dtype=lib.float32), lib.constant, lib


def assert_device(value, lib):
    if REQUIRE_GPU:
        assert value.device.type == "cuda" if hasattr(lib, "autograd") else "GPU:0" in value.device


def array(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x.numpy()


@pytest.mark.parametrize("mode,expected", [("sum", [[4., 6.], [0., 0.]]),
                                           ("mean", [[2., 3.], [0., 0.]]),
                                           ("max", [[3., 4.], [0., 0.]])])
def test_pooling_ignores_nan_padding_and_defines_empty_rows(native, mode, expected):
    m, tensor, boolean, lib = native
    x = tensor([[[1., 2.], [3., 4.], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]])
    mask = boolean([[True, True, False], [False, False, False]])
    out = m.MaskedSequencePooling(mode)(x, valid=mask)
    assert_device(out, lib)
    np.testing.assert_allclose(array(out), expected)


@pytest.mark.parametrize("mode,normalization,expected", [
    ("sum", "none", [[10., 14.]]), ("mean", "none", [[2.5, 3.5]]),
    ("max", "none", [[9., 12.]]), ("sum", "softmax", [[2.5, 3.5]]),
    ("mean", "softmax", [[2.5, 3.5]])])
def test_weighted_pooling_has_declared_normalization(native, mode, normalization, expected):
    m, tensor, boolean, lib = native
    weights = [1., 3., np.nan] if normalization == "none" else [0., math.log(3), np.nan]
    out = m.MaskedSequencePooling(mode, weight_normalization=normalization)(
        tensor([[[1., 2.], [3., 4.], [np.nan, np.nan]]]),
        valid=boolean([[True, True, False]]), weights=tensor([weights]))
    assert_device(out, lib)
    np.testing.assert_allclose(array(out), expected, rtol=1e-6)


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_padding_and_zero_weights_have_finite_zero_gradients(native, mode):
    m, tensor, boolean, lib = native
    x = tensor([[[1., 2.], [np.nan, np.nan]], [[3., 4.], [np.nan, np.nan]]])
    weights = tensor([[2., np.nan], [0., np.nan]])
    mask = boolean([[True, False], [True, False]])
    if hasattr(lib, "autograd"):
        x.requires_grad_(); weights.requires_grad_()
        out = m.MaskedSequencePooling(mode)(x, valid=mask, weights=weights)
        out.sum().backward()
        gradients = [x.grad, weights.grad]
    else:
        with lib.GradientTape() as tape:
            tape.watch([x, weights])
            out = m.MaskedSequencePooling(mode)(x, valid=mask, weights=weights)
            loss = lib.reduce_sum(out)
        gradients = tape.gradient(loss, [x, weights])
    np.testing.assert_array_equal(array(out)[1], 0)
    assert_device(out, lib)
    for gradient in gradients:
        assert_device(gradient, lib)
        assert gradient is not None and np.isfinite(array(gradient)).all()
        np.testing.assert_array_equal(array(gradient)[:, 1], 0)


def test_softmax_empty_rows_and_zero_width_sequences(native):
    m, tensor, boolean, _ = native
    for mode in ("sum", "mean", "max"):
        layer = m.MaskedSequencePooling(mode, weight_normalization="softmax")
        np.testing.assert_array_equal(array(layer(tensor([[[np.nan, np.nan]]]),
            valid=boolean([[False]]), weights=tensor([[np.nan]]))), [[0, 0]])
        np.testing.assert_array_equal(array(layer(tensor(np.empty((2, 0, 3))),
            valid=boolean(np.empty((2, 0), dtype=bool)), weights=tensor(np.empty((2, 0))))), np.zeros((2, 3)))


def test_mask_length_prefix_and_capacity_contract(native):
    m, tensor, boolean, lib = native
    x = tensor(np.zeros((2, 3, 2)))
    lengths = boolean([2, 0])
    expected = [[True, True, False], [False, False, False]]
    np.testing.assert_array_equal(array(m.validate_sequence_mask(x, lengths=lengths)), expected)
    mask = boolean([[True, False, True], [False, False, False]])
    np.testing.assert_array_equal(array(m.validate_sequence_mask(x, mask, lengths)), array(mask))
    errors = (ValueError,) if hasattr(lib, "autograd") else (ValueError, lib.errors.InvalidArgumentError)
    for kwargs in ({"valid": mask, "lengths": boolean([1, 0])},
                   {"valid": mask, "require_prefix": True}, {"lengths": boolean([4, 0])},
                   {"lengths": tensor([2., 0.])}, {"valid": tensor(expected)}, {"max_length": 2}):
        with pytest.raises(errors):
            m.validate_sequence_mask(x, **kwargs)


def test_valid_nan_and_invalid_weights_are_rejected(native):
    m, tensor, boolean, lib = native
    errors = (ValueError,) if hasattr(lib, "autograd") else (ValueError, lib.errors.InvalidArgumentError)
    layer = m.MaskedSequencePooling()
    for x, w in (([[[np.nan]]], [[1.]]), ([[[1.]]], [[np.inf]]), ([[[1.]]], [[-1.]])):
        with pytest.raises(errors):
            layer(tensor(x), valid=boolean([[True]]), weights=tensor(w))
    with pytest.raises(ValueError):
        m.MaskedSequencePooling(weight_normalization="softmax")(tensor([[[1.]]]))


def test_event_alignment_checks_actual_masks_and_event_ids(native):
    m, tensor, boolean, lib = native
    attributes = {"item": tensor([[[1.], [np.nan]]]), "price": tensor([[[3.], [np.nan]]])}
    masks = {key: boolean([[True, False]]) for key in attributes}
    events = {"item": boolean([[10, 0]]), "price": boolean([[10, 99]])}
    m.validate_aligned_sequences(attributes, masks, events)
    errors = (ValueError,) if hasattr(lib, "autograd") else (ValueError, lib.errors.InvalidArgumentError)
    events["price"] = boolean([[11, 99]])
    with pytest.raises(errors):
        m.validate_aligned_sequences(attributes, masks, events)
    masks["price"] = boolean([[True, True]])
    with pytest.raises(errors):
        m.validate_aligned_sequences(attributes, masks)


def test_tensorflow_pooling_works_with_dynamic_shapes():
    tf = runtime("tensorflow")
    m = importlib.import_module("model_evo_harness.models.tensorflow.sequence_inputs")
    with tf.device("/GPU:0" if REQUIRE_GPU else "/CPU:0"):
        layer = m.MaskedSequencePooling("mean", weight_normalization="softmax")
        @tf.function(input_signature=[tf.TensorSpec([None, None, 2], tf.float32),
                                     tf.TensorSpec([None, None], tf.bool),
                                     tf.TensorSpec([None, None], tf.float32)])
        def run(x, valid, weights):
            return layer(x, valid=valid, weights=weights)
        out = run(tf.ones([2, 3, 2]), tf.ones([2, 3], tf.bool), tf.zeros([2, 3]))
        assert_device(out, tf)
        np.testing.assert_array_equal(out, np.ones((2, 2)))
        np.testing.assert_array_equal(run(tf.zeros([2, 0, 2]), tf.zeros([2, 0], tf.bool), tf.zeros([2, 0])), np.zeros((2, 2)))


def test_training_scales_are_hand_computed_and_batch_duplication_invariant(native):
    m, tensor, boolean, lib = native
    training = importlib.import_module(m.__name__.replace("sequence_inputs", "ctr_training"))
    objective = training.NextBehaviorObjective(1, 1, 1)
    if REQUIRE_GPU and hasattr(lib, "autograd"):
        objective = objective.cuda()
    # Constant scorer logit 2 makes the two-label arithmetic independent of weights.
    if hasattr(lib, "autograd"):
        with lib.no_grad():
            for p in objective.parameters(): p.zero_()
            objective.scorer[-1].bias.fill_(2.)
    else:
        for p in objective.trainable_variables: p.assign(lib.zeros_like(p))
        objective.scorer.layers[-1].bias.assign([2.])
    for batch in (1, 3):
        x = tensor(np.zeros((batch, 2, 1)))
        valid = boolean([[True, False]] * batch)
        result = objective(x, x, x, valid)
        np.testing.assert_allclose(array(result["loss"]), (np.logaddexp(0, -2.) + np.logaddexp(0, 2.))/2, rtol=1e-6)
        assert int(array(result["valid_count"])) == batch
    for p in (1, 2):
        result = training.GroupedRegularization(.2, .4, p=p)([tensor([-1., 2.])], [tensor([-3.])])
        np.testing.assert_allclose(array(result["loss"]), .2/p*(1+2**p)+.4/p*3**p, rtol=1e-6)


def test_softmax_pooling_gradient_matches_attention_and_masks_nan_padding(native):
    m, tensor, boolean, lib = native
    x = tensor([[[1.], [3.], [np.nan]], [[np.nan], [np.nan], [np.nan]]])
    logits = tensor([[0., math.log(3), np.nan], [np.nan, np.nan, np.nan]])
    mask = boolean([[True, True, False], [False, False, False]])
    layer = m.MaskedSequencePooling("sum", weight_normalization="softmax")
    if hasattr(lib, "autograd"):
        x.requires_grad_(); logits.requires_grad_()
        out = layer(x, valid=mask, weights=logits)
        out.sum().backward()
        gradients = [x.grad, logits.grad]
    else:
        with lib.GradientTape() as tape:
            tape.watch([x, logits])
            out = layer(x, valid=mask, weights=logits)
            loss = lib.reduce_sum(out)
        gradients = tape.gradient(loss, [x, logits])
    assert_device(out, lib)
    for gradient in gradients:
        assert_device(gradient, lib)
    np.testing.assert_allclose(array(out), [[2.5], [0.]])
    np.testing.assert_allclose(array(gradients[0]), [[[.25], [.75], [0]], [[0], [0], [0]]], rtol=1e-6)
    np.testing.assert_allclose(array(gradients[1]), [[-.375, .375, 0], [0, 0, 0]], rtol=1e-6)


def test_negative_max_values_are_not_replaced_by_padding_zero(native):
    m, tensor, boolean, _ = native
    out = m.MaskedSequencePooling("max")(tensor([[[-3., -4.], [np.nan, np.nan]]]),
                                         valid=boolean([[True, False]]))
    np.testing.assert_array_equal(array(out), [[-3., -4.]])


def test_mask_and_length_shapes_cannot_pass_by_broadcasting(native):
    m, tensor, boolean, lib = native
    x = tensor(np.zeros((2, 2, 1)))
    errors = (ValueError,) if hasattr(lib, "autograd") else (ValueError, lib.errors.InvalidArgumentError)
    for kwargs in ({"lengths": boolean(2)}, {"valid": boolean([True, True])},
                   {"lengths": boolean([[2, 2]])}):
        with pytest.raises(errors):
            m.validate_sequence_mask(x, **kwargs)
    with pytest.raises(errors):
        m.MaskedSequencePooling()(x, weights=tensor([1., 1.]))
