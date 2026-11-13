"""Both native references must compose identically with an existing backbone."""

import numpy as np
import pytest

from model_evo_harness import load_catalog, read_references

torch = pytest.importorskip("torch")
tf = pytest.importorskip("tensorflow")


def _assign_matching_weights(pytorch, tensorflow, attributes):
    rng = np.random.default_rng(19)
    for name in attributes:
        torch_layers = [layer for layer in getattr(pytorch, name).modules()
                        if isinstance(layer, torch.nn.Linear)]
        tf_part = getattr(tensorflow, name)
        tf_layers = tf_part.layers if isinstance(tf_part, tf.keras.Sequential) else [tf_part]
        assert len(torch_layers) == len(tf_layers)
        for pt_layer, tf_layer in zip(torch_layers, tf_layers):
            kernel = rng.normal(0, .2, tuple(pt_layer.weight.shape)).astype(np.float32)
            bias = rng.uniform(.1, .3, tuple(pt_layer.bias.shape)).astype(np.float32)
            with torch.no_grad():
                pt_layer.weight.copy_(torch.from_numpy(kernel))
                pt_layer.bias.copy_(torch.from_numpy(bias))
            tf_layer.kernel.assign(kernel.T)
            tf_layer.bias.assign(bias)


def test_matched_native_components_agree_on_predictions_objective_and_input_gradients():
    from model_evo_harness.models.pytorch.discount_aware import (
        DiscountPreferenceBranch as TorchBranch, DiscountAwareConversionHead as TorchHead,
    )
    from model_evo_harness.models.tensorflow.discount_aware import (
        DiscountPreferenceBranch as TFBranch, DiscountAwareConversionHead as TFHead,
    )

    arrays = [
        [[.1, .3, .2, .6, .4], [.2, .5, np.nan, np.nan, np.nan], [np.nan] * 5],
        [[True] * 5, [True, True, False, False, False], [False] * 5],
        [[.2, .4], [.3, .1], [.1, .2]],
        [[[.3, .5], [.2, .1]], [[.5, .4], [np.nan, np.nan]], [[np.nan, np.nan]] * 2],
        [[[.1, .3, .5], [.2, .4, .1]], [[.2, .6, .1], [np.nan] * 3], [[np.nan] * 3] * 2],
        [[True, True], [True, False], [False, False]],
        [[.5, .1], [.2, .3], [.1, .1]],
    ]
    pt_args = [torch.tensor(value, dtype=torch.bool if i in (1, 5) else torch.float32,
                            requires_grad=i not in (1, 5)) for i, value in enumerate(arrays)]
    tf_args = [tf.constant(value, dtype=tf.bool) if i in (1, 5) else tf.Variable(value, dtype=tf.float32)
               for i, value in enumerate(arrays)]
    interest = [[.2, .5], [.1, .3], [.3, .6]]
    pt_interest, tf_interest = torch.tensor(interest, requires_grad=True), tf.Variable(interest)
    pt_branch, tf_branch = TorchBranch(5, 2, 3, 2, 3), TFBranch(5, 2, 3, 2, 3)
    pt_head, tf_head = TorchHead(2, 13, 3), TFHead(2, 13, 3)
    tf_head(tf_interest, tf_branch(*tf_args)["representation"])
    _assign_matching_weights(pt_branch, tf_branch, (
        "spectral_filter", "attribute_attention", "user_projection", "context_gate"))
    _assign_matching_weights(pt_head, tf_head, ("conversion_head", "discount_head"))

    pt_result = pt_branch(*pt_args)
    pt_outputs = pt_head(pt_interest, pt_result["representation"])
    pt_losses = pt_head.objective(pt_outputs, torch.tensor([1., 0., 1.]),
                                 torch.tensor([.25, np.nan, np.nan]),
                                 torch.tensor([True, False, False]), alpha=.4)
    pt_losses["loss"].backward()
    with tf.GradientTape() as tape:
        tf_result = tf_branch(*tf_args)
        tf_outputs = tf_head(tf_interest, tf_result["representation"])
        tf_losses = tf_head.objective(tf_outputs, tf.constant([1., 0., 1.]),
                                     tf.constant([.25, np.nan, np.nan]),
                                     tf.constant([True, False, False]), alpha=.4)
    indices = (0, 2, 3, 4, 6)
    tf_gradients = tape.gradient(tf_losses["loss"], [tf_args[i] for i in indices] + [tf_interest])
    for pt_mapping, tf_mapping in ((pt_result, tf_result), (pt_outputs, tf_outputs),
                                    (pt_losses, tf_losses)):
        assert pt_mapping.keys() == tf_mapping.keys()
        for name in pt_mapping:
            np.testing.assert_allclose(pt_mapping[name].detach().numpy(), tf_mapping[name].numpy(),
                                       rtol=2e-5, atol=2e-6, err_msg=name)
    for pt_input, tf_gradient in zip([pt_args[i] for i in indices] + [pt_interest], tf_gradients):
        assert tf_gradient is not None and pt_input.grad is not None
        np.testing.assert_allclose(pt_input.grad.numpy(), tf_gradient.numpy(), rtol=3e-5, atol=2e-6)
    assert np.linalg.norm(pt_args[0].grad[0].numpy()) > 0
    assert np.linalg.norm(pt_interest.grad.numpy()) > 0
    np.testing.assert_array_equal(pt_args[0].grad[2].numpy(), 0)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_agent_supplied_source_runs_as_an_independent_module(framework):
    bundle = read_references(load_catalog(), {"framework": framework,
        "method_ids": ["danet_discount_preference", "danet_conversion_head"]})
    namespace = {"__name__": "standalone_discount_components"}
    for path, record in bundle["files"].items():
        exec(compile(record["content"], path, "exec"), namespace)
    if framework == "pytorch":
        ones = lambda shape: torch.ones(shape)
        mask = lambda shape: torch.ones(shape, dtype=torch.bool)
        to_array = lambda value: value.detach().numpy()
    else:
        ones = lambda shape: tf.ones(shape)
        mask = lambda shape: tf.ones(shape, tf.bool)
        to_array = lambda value: value.numpy()
    branch = namespace["DiscountPreferenceBranch"](3, 2, 3, 2, 2)
    head = namespace["DiscountAwareConversionHead"](2, 8, 2)
    features = branch(ones((2, 3)), mask((2, 3)), ones((2, 2)), ones((2, 1, 2)),
                      ones((2, 1, 3)), mask((2, 1)), ones((2, 2)))
    outputs = head(ones((2, 2)), features["representation"])
    losses = head.objective(outputs, ones((2,)), ones((2,)), mask((2,)))
    assert np.isfinite(to_array(losses["loss"]))
    assert to_array(losses["discount_count"]) == 2
