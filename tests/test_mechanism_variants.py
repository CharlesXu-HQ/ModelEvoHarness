"""Explicit structural choices preserve defaults and expose distinct mechanisms."""
import importlib

import numpy as np
import pytest


def api(framework, name):
    return importlib.import_module(f"model_evo_harness.models.{framework}.{name}")


def tensor(framework, values):
    if framework == "pytorch":
        import torch
        return torch.tensor(values)
    import tensorflow as tf
    return tf.constant(values)


def array(value):
    return value.detach().numpy() if hasattr(value, "detach") else value.numpy()


def assign(variable, value):
    if hasattr(variable, "assign"):
        variable.assign(value)
    else:
        import torch
        with torch.no_grad():
            variable.copy_(torch.as_tensor(value, dtype=variable.dtype))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_cin_identity_retains_negative_interactions_and_default_is_relu(framework):
    cls = api(framework, "interactions").XDeepFM
    ids = tensor(framework, [[0, 0]])
    results = []
    for kwargs in ({}, {"cin_activation": "relu"}, {"cin_activation": "identity"}):
        model = cls([1, 1], 1, cin_channels=[1, 1], **kwargs)
        model(ids)
        table = model.fields.embeddings
        assign(table.weight if framework == "pytorch" else table.embeddings, [[2.], [0.]])
        assign(model.cin_weights[0], [[-1., 0., 0., 0.]])
        assign(model.cin_weights[1], [[1., 0.]])
        results.append(array(model.cin_features(ids)))
    np.testing.assert_array_equal(results[0], [[0., 0.]])
    np.testing.assert_array_equal(results[1], results[0])
    # x0=2; layer 1=-x0*x0=-4; layer 2=x0*(-4)=-8.
    np.testing.assert_array_equal(results[2], [[-4., -8.]])
    with pytest.raises(ValueError):
        cls([1, 1], 1, cin_activation="unrecognized")


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_sdim_l2_is_applied_per_hash_before_averaging(framework):
    cls = api(framework, "history_retrieval").SDIMCollisionPooling
    target = tensor(framework, [[3., 1.]])
    history = tensor(framework, [[[1., 2.], [3., 4.], [4., 1.], [-1., 2.]]])
    valid = tensor(framework, [[True] * 4])
    rotations = [[[1., 0.], [0., 1.]], [[1., 1.], [1., -2.]]]
    results = []
    for kwargs in ({}, {"normalize_per_hash": False}, {"normalize_per_hash": True}):
        model = cls(2, 2, 2, **kwargs)
        assign(model.rotations, rotations)
        results.append(model(target, history, valid))
    np.testing.assert_allclose(array(results[0]["representation"]), [[6., 4.]])
    np.testing.assert_array_equal(array(results[1]["representation"]), array(results[0]["representation"]))
    # Raw bucket sums remain observable under either option.
    np.testing.assert_array_equal(array(results[2]["per_hash"]), [[[8., 7.], [4., 1.]]])
    expected = (np.array([8., 7.]) / np.sqrt(113) + np.array([4., 1.]) / np.sqrt(17)) / 2
    np.testing.assert_allclose(array(results[2]["representation"]), [expected], rtol=1e-6)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
@pytest.mark.parametrize("empty", [False, True])
def test_normalized_sdim_zero_buckets_keep_finite_gradients(framework, empty):
    cls = api(framework, "history_retrieval").SDIMCollisionPooling
    model = cls(2, 1, 2, normalize_per_hash=True)
    assign(model.rotations, [[[1., 0.], [0., 1.]]])
    target = tensor(framework, [[1., 1.]])
    values = np.empty((1, 0, 2), np.float32) if empty else np.array([[[0., 0.], [np.nan, np.nan]]], np.float32)
    mask = tensor(framework, np.empty((1, 0), bool) if empty else [[True, False]])
    if framework == "pytorch":
        import torch
        history = torch.tensor(values, requires_grad=True)
        output = model(target, history, mask)["representation"]
        output.sum().backward()
        gradients = history.grad
    else:
        import tensorflow as tf
        history = tf.Variable(values)
        with tf.GradientTape() as tape:
            output = model(target, history, mask)["representation"]
            loss = tf.reduce_sum(output)
        gradients = tape.gradient(loss, history)
    np.testing.assert_array_equal(array(output), [[0., 0.]])
    assert gradients is not None
    assert np.isfinite(array(gradients)).all()
    if not empty:
        np.testing.assert_array_equal(array(gradients)[:, 1], 0.)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
@pytest.mark.parametrize("magnitude", [0., 1e-7])
def test_normalized_sdim_half_precision_small_buckets_have_finite_gradients(framework, magnitude):
    cls = api(framework, "history_retrieval").SDIMCollisionPooling
    if framework == "pytorch":
        import torch
        model = cls(2, 1, 2, normalize_per_hash=True).half()
        assign(model.rotations, [[[1., 0.], [0., 1.]]])
        history = torch.tensor([[[magnitude, 0.], [np.nan, np.nan]]], dtype=torch.float16, requires_grad=True)
        output = model(torch.ones((1, 2), dtype=torch.float16), history, torch.tensor([[True, False]]))["representation"]
        output.sum().backward()
        gradient = history.grad
    else:
        import tensorflow as tf
        old_policy = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            model = cls(2, 1, 2, normalize_per_hash=True)
            assign(model.rotations, [[[1., 0.], [0., 1.]]])
            history = tf.Variable([[[magnitude, 0.], [np.nan, np.nan]]], dtype=tf.float16)
            with tf.GradientTape() as tape:
                output = model(tf.ones((1, 2), tf.float16), history, tf.constant([[True, False]]))["representation"]
                loss = tf.reduce_sum(output)
            gradient = tape.gradient(loss, history)
        finally:
            tf.keras.mixed_precision.set_global_policy(old_policy)
    assert array(output).dtype == np.float16
    assert np.isfinite(array(output)).all()
    assert np.isfinite(array(gradient)).all()
    np.testing.assert_array_equal(array(gradient)[:, 1], 0.)
