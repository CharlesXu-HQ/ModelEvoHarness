"""TensorFlow execution contracts for the single-view QueryFormer bridge."""

import importlib
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models/tensorflow/paper_queryformer.py"
    assert path.is_file(), "Missing TensorFlow QueryFormer implementation"
    return importlib.import_module("model_evo_harness.models.tensorflow.paper_queryformer")


def test_bridge_accepts_dynamic_batch_and_token_lengths_in_one_graph():
    import tensorflow as tf

    model = _module().QueryFormerQueryBridge(4, 2, 2)
    tokens = tf.TensorSpec([None, None, 4], tf.float32)
    mask = tf.TensorSpec([None, None], tf.bool)

    @tf.function(input_signature=[tokens, tokens, tokens, tokens, mask, mask, mask, mask])
    def run(user, item, first, second, um, im, fm, sm):
        return model(user, item, [first, second], um, im, [fm, sm])

    for batch, user_len, item_len, first_len, second_len in [(2, 3, 2, 4, 1), (1, 0, 0, 0, 0), (3, 1, 0, 2, 0)]:
        lengths = (user_len, item_len, first_len, second_len)
        values = [tf.ones([batch, length, 4]) for length in lengths]
        masks = [tf.ones([batch, length], tf.bool) for length in lengths]
        output = run(*values, *masks)
        assert output["queries"].shape == (batch, 2, 2, 4)
        assert output["context_tokens"].shape == (batch, user_len + item_len + 2, 4)
        assert np.isfinite(output["queries"].numpy()).all()
        expected = np.array([first_len > 0, second_len > 0]) & (user_len + item_len > 0)
        np.testing.assert_array_equal(output["query_mask"], np.broadcast_to(expected[None, :, None], (batch, 2, 2)))
        if user_len + item_len == 0:
            np.testing.assert_array_equal(output["queries"], np.zeros((batch, 2, 2, 4)))


def test_bridge_ignores_nonfinite_padding_and_keeps_absent_domains_zero():
    import tensorflow as tf

    model = _module().QueryFormerQueryBridge(4, 2, 2)
    rng = np.random.default_rng(18)
    values = [rng.normal(size=(2, length, 4)).astype("float32") for length in [2, 2, 3, 2]]
    masks = [np.array([[True] + [False] * (x.shape[1] - 1), [False] * x.shape[1]]) for x in values]
    masks[-1][:] = False

    def run(data):
        tensors = [tf.constant(x) for x in data]
        valid = [tf.constant(x) for x in masks]
        return model(tensors[0], tensors[1], tensors[2:], valid[0], valid[1], valid[2:])

    run(values)
    # Learned affine offsets must not manufacture tokens for missing evidence.
    for variable in model.trainable_variables:
        if variable.name in {"bias", "beta"}:
            variable.assign(tf.ones_like(variable))
    clean = run(values)
    padded = [np.where(mask[..., None], value, np.nan) for value, mask in zip(values, masks)]
    dirty = run(padded)
    for name in ["queries", "context_tokens"]:
        np.testing.assert_allclose(clean[name], dirty[name], atol=1e-6)
        assert np.isfinite(dirty[name].numpy()).all()
        np.testing.assert_array_equal(dirty[name][1], np.zeros(dirty[name][1].shape))
    np.testing.assert_array_equal(dirty["queries"][:, 1], np.zeros((2, 2, 4)))
    np.testing.assert_array_equal(dirty["context_tokens"][:, -1], np.zeros((2, 4)))
    np.testing.assert_array_equal(dirty["query_mask"], [[[True, True], [False, False]], [[False, False], [False, False]]])


def test_attention_handles_zero_length_queries_and_keys():
    import tensorflow as tf

    layer = _module()._Attention(4, 2)
    for query_len, key_len in [(2, 0), (0, 2), (0, 0)]:
        result = layer(tf.ones([2, query_len, 4]), tf.ones([2, key_len, 4]),
                       tf.ones([2, query_len], tf.bool), tf.ones([2, key_len], tf.bool))
        np.testing.assert_array_equal(result, np.zeros((2, query_len, 4)))


@pytest.mark.parametrize("bad_input", ["mask_dtype", "mask_rank", "mask_shape", "token_rank", "width", "batch", "domains"])
def test_bridge_rejects_invalid_input_contracts(bad_input):
    import tensorflow as tf

    model = _module().QueryFormerQueryBridge(4, 2, 1)
    user = item = history = tf.ones([2, 3, 4])
    um = im = hm = tf.ones([2, 3], tf.bool)
    model(user, item, [history], um, im, [hm])
    if bad_input == "mask_dtype":
        um = tf.ones([2, 3], tf.int32)
    elif bad_input == "mask_rank":
        um = tf.ones([2, 3, 1], tf.bool)
    elif bad_input == "mask_shape":
        um = tf.ones([2, 2], tf.bool)
    elif bad_input == "token_rank":
        user = tf.ones([2, 4])
    elif bad_input == "width":
        history = tf.ones([2, 3, 5])
    elif bad_input == "batch":
        history, hm = tf.ones([1, 3, 4]), tf.ones([1, 3], tf.bool)
    with pytest.raises((ValueError, TypeError, tf.errors.InvalidArgumentError)):
        model(user, item, [] if bad_input == "domains" else [history], um, im, [hm])


def test_mixed_float16_attention_accumulates_without_overflow():
    import tensorflow as tf

    module = _module()
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        layer = module._Attention(4, 2)
        query = tf.Variable([[[1000., 1000., 1000., 1000.]]], dtype=tf.float16)
        keys = tf.Variable([[[1000., 1000., 1000., 1000.], [np.nan] * 4]], dtype=tf.float16)
        qm, km = tf.constant([[True]]), tf.constant([[True, False]])
        layer(query, keys, qm, km)
        for projection in [layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj]:
            projection.kernel.assign(tf.eye(4))
            projection.bias.assign(tf.zeros([4]))
        with tf.GradientTape() as tape:
            result = layer(query, keys, qm, km)
            loss = tf.reduce_sum(tf.cast(result, tf.float32))
        gradients = tape.gradient(loss, [query, keys] + layer.trainable_variables)
        assert result.dtype == tf.float16
        np.testing.assert_array_equal(result, [[[1000.] * 4]])
        assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
        np.testing.assert_array_equal(gradients[1][0, 1], [0.] * 4)
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)


def test_mixed_float16_bridge_has_finite_gradients_with_nan_padding():
    import tensorflow as tf

    module = _module()
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = module.QueryFormerQueryBridge(4, 2, 2)
        mask = tf.constant([[True, True, False], [False, False, False]])
        rng = np.random.default_rng(24)
        inputs = [tf.Variable(np.where(mask.numpy()[..., None], rng.normal(size=(2, 3, 4)), np.nan), dtype=tf.float16) for _ in range(4)]
        with tf.GradientTape() as tape:
            result = model(inputs[0], inputs[1], inputs[2:], mask, mask, [mask, mask])
            loss = tf.reduce_sum(tf.square(tf.cast(result["queries"], tf.float32)))
        gradients = tape.gradient(loss, inputs + model.trainable_variables)
        assert result["queries"].dtype == tf.float16
        assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
        for gradient in gradients[:4]:
            np.testing.assert_array_equal(gradient.numpy()[~mask.numpy()], np.zeros((4, 4)))
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
