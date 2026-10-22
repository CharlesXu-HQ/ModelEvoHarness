"""Numeric attention directions, masks and gradients for a single-view query bridge."""

import importlib
import math
from pathlib import Path

import numpy as np
import pytest


def _module(framework):
    path = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models" / framework / "paper_queryformer.py"
    assert path.is_file(), f"Missing QueryFormer tensor module: {path}"
    return importlib.import_module(f"model_evo_harness.models.{framework}.paper_queryformer")


def _tensor(framework, value):
    value = np.asarray(value)
    if value.dtype != bool:
        value = value.astype("float32")
    if framework == "pytorch":
        import torch
        return torch.tensor(value)
    import tensorflow as tf
    return tf.constant(value)


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value.numpy()


def _linear(layer, kernel, bias):
    if hasattr(layer, "kernel"):
        layer.kernel.assign(np.asarray(kernel, np.float32))
        layer.bias.assign(np.asarray(bias, np.float32))
    else:
        import torch
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(np.asarray(kernel).T, dtype=layer.weight.dtype))
            layer.bias.copy_(torch.tensor(bias, dtype=layer.bias.dtype))


def _identity(attention, dim):
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        _linear(getattr(attention, name), np.eye(dim), np.zeros(dim))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_history_retrieval_uses_item_mean_and_preserves_domain_order(framework):
    model = _module(framework).QuerySeqCrossAttn(2, 1, 2)
    t = lambda v: _tensor(framework, v)
    items = t([[[2, 0], [0, 0], [np.nan, np.inf]]])
    history = t([[[1, 0], [0, 2], [np.nan, np.inf]]])
    masks = [t([[True, True, False]]), t([[False, False, False]])]
    args = (items, [history, history], t([[True, True, False]]), masks)
    model(*args)
    for attention in model.attentions:
        _identity(attention, 2)
    output, valid = model(*args)
    p = 1 / (1 + math.exp(-1 / math.sqrt(2)))
    np.testing.assert_allclose(_array(output), [[[p, 2*(1-p)], [0, 0]]], atol=1e-6)
    np.testing.assert_array_equal(_array(valid), [[True, False]])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_mean_max_probes_query_context_not_history(framework):
    model = _module(framework).SeqQueryCrossAttn(2, 1, 1)
    t = lambda v: _tensor(framework, v)
    context = t([[[1, 0], [0, 1], [np.nan, np.inf]]])
    history = t([[[2, 0], [0, 1], [np.nan, np.inf]]])
    args = (context, [history], t([[True, True, False]]), [t([[True, True, False]])])
    model(*args)
    _identity(model.attentions[0], 2)
    output, valid = model(*args)
    mean_p = 1 / (1 + math.exp(-.5 / math.sqrt(2)))
    max_p = 1 / (1 + math.exp(-1 / math.sqrt(2)))
    np.testing.assert_allclose(_array(output), [[[[mean_p, 1-mean_p], [max_p, 1-max_p]]]], atol=1e-6)
    np.testing.assert_array_equal(_array(valid), [[[True, True]]])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_co_transformer_self_stage_then_user_to_item_gated_cross(framework):
    model = _module(framework).CoTransformerBlock(3, 1, num_layers=1)
    t = lambda v: _tensor(framework, v)
    args = (t([[[1, 2, 4]]]), t([[[4, 0, -1]]]), t([[True]]), t([[True]]))
    model(*args)
    for group in (model.user_self, model.item_self, model.user_cross):
        _identity(group[0], 3)
    _linear(model.user_gate[0], np.zeros((6, 3)), np.zeros(3))
    users, items = model(*args)
    # One-token identity attention gives LN(2*x), then LN(user + .5*item).
    np.testing.assert_allclose(_array(users), [[[-0.60464876, -0.80481646, 1.40946522]]], atol=2e-5)
    np.testing.assert_allclose(_array(items), [[[1.38872978, -0.46290993, -0.92581985]]], atol=2e-5)
    # Item states must not acquire a reverse user->item update absent from Table 2.
    changed_users = t([[[9, -3, 7]]])
    _, same_items = model(changed_users, *args[1:])
    np.testing.assert_array_equal(_array(items), _array(same_items))


def _inputs(framework, poisoned=False):
    rng = np.random.default_rng(29)
    values = [rng.normal(size=s).astype("float32") for s in ((2, 2, 4), (2, 2, 4), (2, 3, 4))]
    masks = [np.array([[True, False], [False, False]]), np.array([[True, True], [False, False]]),
             np.array([[True, False, True], [False, False, False]])]
    if poisoned:
        for value, mask in zip(values, masks):
            value[~mask] = np.nan
    u, i, h = [_tensor(framework, x) for x in values]
    um, im, hm = [_tensor(framework, x) for x in masks]
    empty = _tensor(framework, np.zeros((2, 0, 4)))
    em = _tensor(framework, np.zeros((2, 0), bool))
    return u, i, [h, empty], um, im, [hm, em]


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_bridge_context_order_and_empty_domain_masks_survive_biases(framework):
    model = _module(framework).QueryFormerQueryBridge(4, 2, 2)
    args = _inputs(framework)
    clean = model(*args)
    dirty = model(*_inputs(framework, poisoned=True))
    for key in clean:
        np.testing.assert_allclose(_array(clean[key]), _array(dirty[key]), atol=1e-6)
    np.testing.assert_array_equal(_array(dirty["context_mask"]),
                                  [[True, False, True, True, True, False], [False]*6])
    np.testing.assert_array_equal(_array(dirty["query_mask"]),
                                  [[[True, True], [False, False]], [[False, False], [False, False]]])
    for key in ("queries", "context_tokens"):
        assert np.isfinite(_array(dirty[key])).all()
        np.testing.assert_array_equal(_array(dirty[key])[1], 0)
    u, i = model.co_transformer(args[0], args[1], args[3], args[4])
    retrieved, _ = model.retrieval(i, args[2], args[4], args[5])
    expected = np.concatenate([_array(u), _array(i), _array(retrieved)], 1)
    np.testing.assert_allclose(_array(clean["context_tokens"]), expected, atol=1e-6)
    # Nonzero output biases may not fabricate missing context or domain evidence.
    for attention in list(model.retrieval.attentions) + list(model.query_generator.attentions):
        _linear(attention.out_proj, np.eye(4), np.ones(4)*7)
    result = model(*args)
    np.testing.assert_array_equal(_array(result["queries"])[1], 0)
    np.testing.assert_array_equal(_array(result["queries"])[:, 1], 0)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_zero_length_fields_and_histories_return_finite_empty_results(framework):
    model = _module(framework).QueryFormerQueryBridge(4, 2, 1)
    t = lambda v: _tensor(framework, v)
    tokens, mask = t(np.zeros((2, 0, 4))), t(np.zeros((2, 0), bool))
    result = model(tokens, tokens, [tokens], mask, mask, [mask])
    assert tuple(result["context_tokens"].shape) == (2, 1, 4)
    np.testing.assert_array_equal(_array(result["queries"]), 0)
    np.testing.assert_array_equal(_array(result["context_tokens"]), 0)
    np.testing.assert_array_equal(_array(result["query_mask"]), False)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_queryformer_rejects_nonbool_masks_and_wrong_domain_count(framework):
    module = _module(framework)
    with pytest.raises(ValueError):
        module.CoTransformerBlock(5, 2)
    model = module.QuerySeqCrossAttn(4, 2, 2)
    u, i, histories, um, im, masks = _inputs(framework)
    model(i, histories, im, masks)
    with pytest.raises((ValueError, TypeError)):
        model(i, histories[:1], im, masks[:1])
    with pytest.raises((ValueError, TypeError)):
        model(i, histories, _tensor(framework, np.ones((2, 2))), masks)


def test_pytorch_bridge_gradient_flow_and_masked_nan_padding():
    import torch
    model = _module("pytorch").QueryFormerQueryBridge(4, 2, 2)
    u, i, histories, um, im, masks = _inputs("pytorch", poisoned=True)
    u.requires_grad_(); i.requires_grad_(); histories[0].requires_grad_()
    result = model(u, i, histories, um, im, masks)
    loss = (result["queries"] * torch.arange(1, 5)).sum() + (result["context_tokens"] * torch.arange(4, 0, -1)).sum()
    loss.backward()
    for value, mask in zip((u, i, histories[0]), (um, im, masks[0])):
        assert torch.isfinite(value.grad).all()
        assert value.grad[mask].abs().sum() > 0
        assert torch.count_nonzero(value.grad[~mask]) == 0
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_pytorch_half_precision_attention_accumulates_without_score_overflow():
    import torch
    model = _module("pytorch").QuerySeqCrossAttn(4, 2, 1).half()
    _identity(model.attentions[0], 4)
    items = torch.full((2, 1, 4), 300., dtype=torch.float16, requires_grad=True)
    history = torch.full((2, 2, 4), 300., dtype=torch.float16, requires_grad=True)
    valid = torch.tensor([[True, True], [False, False]])
    with torch.no_grad():
        history[1] = float("nan")
    values, _ = model(items, [history], torch.ones((2, 1), dtype=torch.bool), [valid])
    values.sum().backward()
    np.testing.assert_allclose(_array(values)[0], 300.)
    np.testing.assert_array_equal(_array(values)[1], 0)
    for gradient in [items.grad, history.grad] + [p.grad for p in model.parameters()]:
        assert gradient is not None and torch.isfinite(gradient).all()


def test_same_weight_bridge_outputs_and_input_gradients_match_frameworks():
    import torch
    import tensorflow as tf
    pt = _module("pytorch").QueryFormerQueryBridge(4, 2, 2)
    keras = _module("tensorflow").QueryFormerQueryBridge(4, 2, 2)
    keras(*_inputs("tensorflow"))

    def linear(a, b):
        _linear(b, _array(a.weight).T, _array(a.bias))

    def attention(a, b):
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(getattr(a, name), getattr(b, name))

    for name in ("user_self", "item_self", "user_cross"):
        for a, b in zip(getattr(pt.co_transformer, name), getattr(keras.co_transformer, name)):
            attention(a, b)
    for a, b in zip(pt.co_transformer.user_gate, keras.co_transformer.user_gate):
        linear(a, b)
    for name in ("user_self_norm", "item_self_norm", "user_cross_norm"):
        for a, b in zip(getattr(pt.co_transformer, name), getattr(keras.co_transformer, name)):
            b.gamma.assign(_array(a.weight)); b.beta.assign(_array(a.bias))
    for name in ("retrieval", "query_generator"):
        for a, b in zip(getattr(pt, name).attentions, getattr(keras, name).attentions):
            attention(a, b)

    u, i, histories, um, im, masks = _inputs("pytorch")
    variables = [u.requires_grad_(), i.requires_grad_(), histories[0].requires_grad_()]
    output = pt(u, i, histories, um, im, masks)
    weights = torch.tensor([1., 3., -2., 4.])
    ((output["queries"] * weights).sum() + (output["context_tokens"] * weights).sum()).backward()
    tf_u, tf_i, tf_h, tf_um, tf_im, tf_masks = _inputs("tensorflow")
    tf_variables = [tf.Variable(x) for x in (tf_u, tf_i, tf_h[0])]
    with tf.GradientTape() as tape:
        other = keras(tf_variables[0], tf_variables[1], [tf_variables[2], tf_h[1]], tf_um, tf_im, tf_masks)
        loss = tf.reduce_sum(other["queries"] * _array(weights)) + tf.reduce_sum(other["context_tokens"] * _array(weights))
    gradients = tape.gradient(loss, tf_variables)
    for key in output:
        np.testing.assert_allclose(_array(output[key]), _array(other[key]), rtol=3e-5, atol=3e-6)
    for a, b in zip(variables, gradients):
        assert a.grad is not None and b is not None
        np.testing.assert_allclose(_array(a.grad), _array(b), rtol=5e-4, atol=3e-5)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_query_bridge_gpu_forward_backward(framework):
    if framework == "pytorch":
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU unavailable")
        with torch.device("cuda"):
            model = _module(framework).QueryFormerQueryBridge(4, 2, 2)
            result = model(*_inputs(framework))
            result["queries"].square().sum().backward()
            for p in model.parameters():
                assert p.grad is not None and p.grad.is_cuda and torch.isfinite(p.grad).all()
    else:
        import tensorflow as tf
        if not tf.config.list_physical_devices("GPU"):
            pytest.skip("TensorFlow GPU unavailable")
        with tf.device("/GPU:0"):
            model = _module(framework).QueryFormerQueryBridge(4, 2, 2)
            with tf.GradientTape() as tape:
                result = model(*_inputs(framework))
                loss = tf.reduce_sum(tf.square(result["queries"]))
            for gradient in tape.gradient(loss, model.trainable_variables):
                assert gradient is not None and "GPU" in gradient.device
                assert bool(tf.reduce_all(tf.math.is_finite(gradient)))
