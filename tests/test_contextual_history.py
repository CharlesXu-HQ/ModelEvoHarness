"""Context-bearing events and independent, masked relation reads."""

import importlib
import math
from pathlib import Path

import numpy as np
import pytest


RELATIONS = ("user_history", "target_history", "target_user")


def _module(framework):
    path = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models" / framework / "contextual_history.py"
    assert path.is_file(), f"Missing contextual-history tensor module: {path}"
    return importlib.import_module(f"model_evo_harness.models.{framework}.contextual_history")


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


def _linear(layer, kernel, bias=None):
    if hasattr(layer, "kernel"):
        layer.kernel.assign(np.asarray(kernel, np.float32))
        if layer.bias is not None:
            layer.bias.assign(np.asarray(bias, np.float32))
    else:
        import torch
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(np.asarray(kernel).T, dtype=layer.weight.dtype))
            if layer.bias is not None:
                layer.bias.copy_(torch.tensor(bias, dtype=layer.bias.dtype))


def _identity_relations(model):
    for name in RELATIONS:
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            _linear(getattr(getattr(model, name), projection), np.eye(2))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_encoder_uses_event_context_retains_order_and_zeros_padding(framework):
    module = _module(framework)
    t = lambda x: _tensor(framework, x)
    model = module.ContextualHistoryEncoder(2, 1, 2)
    items = t([[[1, 0], [1, 0], [np.nan, np.inf]]])
    contexts = t([[[0], [1], [np.nan]]])
    mask = t([[True, True, False]])
    model(items, contexts, mask)
    _linear(model.item_proj, np.eye(2), [0, 0])
    _linear(model.context_proj, [[1, -1]])
    output, valid = model(items, contexts, mask)
    np.testing.assert_allclose(_array(output), [[[0.76159416, 0], [0.96402758, -0.76159416], [0, 0]]], atol=1e-6)
    np.testing.assert_array_equal(_array(valid), [[True, True, False]])
    # Equal items with different event context must remain distinct in their slots.
    reordered, _ = model(t(_array(items)[:, [1, 0, 2]]), t(_array(contexts)[:, [1, 0, 2]]), mask)
    np.testing.assert_array_equal(_array(reordered), _array(output)[:, [1, 0, 2]])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_relations_have_separate_normalizers_projections_and_known_values(framework):
    model = _module(framework).RelationHistoryAttention(2)
    t = lambda x: _tensor(framework, x)
    args = (t([[[1, 0], [0, 1]]]), t([[[1, 0], [0, 2], [np.nan, np.inf]]]),
            t([[[0, 1], [1, 0], [np.nan, np.inf]]]),
            t([[True, True]]), t([[True, True, False]]), t([[True, True, False]]))
    model(*args)
    _identity_relations(model)
    _linear(model.target_user.v_proj, 3 * np.eye(2))
    result = model(*args, return_attention=True)
    p = 1 / (1 + math.exp(-1 / math.sqrt(2)))
    q = 1 / (1 + math.exp(2 / math.sqrt(2)))
    np.testing.assert_allclose(_array(result["user_history"]), [[[p, 2*(1-p)], [q, 2*(1-q)]]], atol=1e-6)
    np.testing.assert_allclose(_array(result["target_history"]), [[[q, 2*(1-q)], [p, 2*(1-p)], [0, 0]]], atol=1e-6)
    np.testing.assert_allclose(_array(result["target_user"]), [[[3*(1-p), 3*p], [3*p, 3*(1-p)], [0, 0]]], atol=1e-6)
    for name in RELATIONS:
        weights = _array(result["attention"][name])
        expected_mask = [[True, True]] if name == "user_history" else [[True, True, False]]
        np.testing.assert_array_equal(_array(result[name + "_mask"]), expected_mask)
        np.testing.assert_allclose(weights.sum(-1), np.asarray(expected_mask)[:, None], atol=1e-6)
        if name != "target_user":
            np.testing.assert_array_equal(weights[..., -1], 0)
    assert "attention" not in model(*args)


def _inputs(framework):
    rng = np.random.default_rng(71)
    values = [rng.normal(size=(2, length, 4)) for length in (2, 3, 3)]
    masks = [np.array([[True, True], [False, False]]),
             np.array([[True, True, False], [False, False, False]]),
             np.array([[True, True, False], [True, False, False]])]
    values = [np.where(mask[..., None], value, np.nan) for value, mask in zip(values, masks)]
    return tuple(_tensor(framework, x) for x in values + masks)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_evidence_read_only_and_candidates_independent_across_tokens_and_batches(framework):
    model = _module(framework).RelationHistoryAttention(4, 2)
    args = _inputs(framework)
    result = model(*args, return_attention=True)
    changed_target = list(args)
    changed_target[2] = _tensor(framework, _array(args[2]) + 4)
    changed = model(*changed_target)
    np.testing.assert_array_equal(_array(result["user_history"]), _array(changed["user_history"]))
    for batch in range(2):
        for target in range(3):
            one = [x[batch:batch+1] for x in args]
            one[2], one[5] = one[2][:, target:target+1], one[5][:, target:target+1]
            single = model(*one)
            for name in ("target_history", "target_user"):
                np.testing.assert_allclose(_array(single[name]), _array(result[name])[batch:batch+1, target:target+1], atol=1e-6)
    changed_history = list(args)
    changed_history[1] = _tensor(framework, _array(args[1]) + 9)
    np.testing.assert_array_equal(_array(model(*changed_history)["target_user"]), _array(result["target_user"]))
    changed_user = list(args)
    changed_user[0] = _tensor(framework, _array(args[0]) + 9)
    np.testing.assert_array_equal(_array(model(*changed_user)["target_history"]), _array(result["target_history"]))
    for name in RELATIONS:
        assert np.isfinite(_array(result[name])).all()
        np.testing.assert_array_equal(_array(result[name])[1], 0)
        np.testing.assert_array_equal(_array(result["attention"][name])[1], 0)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
@pytest.mark.parametrize("lengths", [(2, 0, 3), (0, 2, 3), (2, 3, 0), (0, 0, 0)])
def test_zero_length_inputs_return_shaped_finite_outputs(framework, lengths):
    module = _module(framework)
    t = lambda x: _tensor(framework, x)
    model = module.RelationHistoryAttention(4, 2)
    values = [t(np.ones((2, length, 4))) for length in lengths]
    masks = [t(np.ones((2, length), bool)) for length in lengths]
    result = model(*values, *masks, return_attention=True)
    for name, q, k in zip(RELATIONS, (lengths[0], lengths[2], lengths[2]), (lengths[1], lengths[1], lengths[0])):
        assert tuple(result[name].shape) == (2, q, 4)
        assert tuple(result["attention"][name].shape) == (2, 2, q, k)
        assert np.isfinite(_array(result[name])).all()
        if not k:
            np.testing.assert_array_equal(_array(result[name]), 0)
            np.testing.assert_array_equal(_array(result[name + "_mask"]), False)
    encoder = module.ContextualHistoryEncoder(4, 2, 4)
    encoded, mask = encoder(values[1], t(np.zeros((2, lengths[1], 2))), masks[1])
    assert tuple(encoded.shape) == (2, lengths[1], 4)
    np.testing.assert_array_equal(_array(mask), _array(masks[1]))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_explicit_dimensions_masks_and_batch_alignment_are_validated(framework):
    module = _module(framework)
    errors = (TypeError, ValueError)
    if framework == "tensorflow":
        import tensorflow as tf
        errors += (tf.errors.InvalidArgumentError,)
    with pytest.raises(ValueError):
        module.RelationHistoryAttention(3, 2)
    with pytest.raises(ValueError):
        module.ContextualHistoryEncoder(2, 0, 4)
    t = lambda x: _tensor(framework, x)
    model = module.RelationHistoryAttention(4, 2)
    args = _inputs(framework)
    model(*args)
    for index, replacement in [(0, t(np.zeros((2, 2, 3)))), (1, t(np.zeros((2, 3)))),
                                (3, t(np.ones((2, 2)))), (4, t(np.ones((2, 2), bool))),
                                (5, t(np.ones((2, 3, 1), bool)))]:
        bad = list(args)
        bad[index] = replacement
        with pytest.raises(errors):
            model(*bad)
    bad = list(args)
    bad[1], bad[4] = bad[1][:1], bad[4][:1]
    with pytest.raises(errors):
        model(*bad)
    encoder = module.ContextualHistoryEncoder(4, 2, 4)
    encoder(args[1], t(np.zeros((2, 3, 2))), args[4])
    for context in (np.zeros((2, 2, 2)), np.zeros((2, 3, 3)), np.zeros((1, 3, 2))):
        with pytest.raises(errors):
            encoder(args[1], t(context), args[4])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_composed_gradients_are_finite_and_padding_has_zero_gradient(framework):
    module = _module(framework)
    args = _inputs(framework)
    encoder = module.ContextualHistoryEncoder(4, 2, 4)
    relation = module.RelationHistoryAttention(4, 2)
    context = np.array([[[.1, .3], [.2, -.4], [np.nan, np.nan]], [[np.nan, np.nan]] * 3], np.float32)
    if framework == "pytorch":
        import torch
        values = [x.clone().requires_grad_() for x in args[:3]] + [torch.tensor(context, requires_grad=True)]
        encoded, hm = encoder(values[1], values[3], args[4])
        result = relation(values[0], encoded, values[2], args[3], hm, args[5])
        loss = sum(result[name].square().sum() for name in RELATIONS)
        loss.backward()
        grads = [x.grad for x in values]
        parameter_grads = [p.grad for p in list(encoder.parameters()) + list(relation.parameters())]
    else:
        import tensorflow as tf
        values = [tf.Variable(x) for x in args[:3]] + [tf.Variable(context)]
        with tf.GradientTape() as tape:
            encoded, hm = encoder(values[1], values[3], args[4])
            result = relation(values[0], encoded, values[2], args[3], hm, args[5])
            loss = sum(tf.reduce_sum(tf.square(result[name])) for name in RELATIONS)
        parameters = encoder.trainable_variables + relation.trainable_variables
        gradients = tape.gradient(loss, values + parameters)
        grads, parameter_grads = gradients[:4], gradients[4:]
    for gradient, mask in zip(grads, (*args[3:], args[4])):
        assert gradient is not None and np.isfinite(_array(gradient)).all()
        assert np.abs(_array(gradient)[_array(mask)]).sum() > 0
        np.testing.assert_array_equal(_array(gradient)[~_array(mask)], 0)
    assert all(g is not None and np.isfinite(_array(g)).all() for g in parameter_grads)


def test_assigned_weights_match_between_native_frameworks():
    modules = [_module(framework) for framework in ("pytorch", "tensorflow")]
    encoders = [module.ContextualHistoryEncoder(4, 2, 4) for module in modules]
    relations = [module.RelationHistoryAttention(4, 2) for module in modules]
    args = [_inputs(framework) for framework in ("pytorch", "tensorflow")]
    contexts = [_tensor(framework, np.ones((2, 3, 2))) for framework in ("pytorch", "tensorflow")]
    for encoder, relation, inputs, context in zip(encoders, relations, args, contexts):
        encoder(inputs[1], context, inputs[4])
        relation(*inputs)
    rng = np.random.default_rng(118)
    pairs = [(encoders[0].item_proj, encoders[1].item_proj, 4, True),
             (encoders[0].context_proj, encoders[1].context_proj, 2, False)]
    for name in RELATIONS:
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            pairs.append((getattr(getattr(relations[0], name), projection),
                          getattr(getattr(relations[1], name), projection), 4, False))
    for first, second, width, bias in pairs:
        kernel, offset = rng.normal(size=(width, 4)) * .2, rng.normal(size=4) * .1 if bias else None
        _linear(first, kernel, offset)
        _linear(second, kernel, offset)
    outputs = []
    for encoder, relation, inputs, context in zip(encoders, relations, args, contexts):
        history, hm = encoder(inputs[1], context, inputs[4])
        outputs.append((history, relation(inputs[0], history, inputs[2], inputs[3], hm, inputs[5], return_attention=True)))
    np.testing.assert_allclose(_array(outputs[0][0]), _array(outputs[1][0]), atol=1e-6)
    for name in RELATIONS:
        np.testing.assert_allclose(_array(outputs[0][1][name]), _array(outputs[1][1][name]), atol=1e-6)
        np.testing.assert_allclose(_array(outputs[0][1]["attention"][name]), _array(outputs[1][1]["attention"][name]), atol=1e-6)


def test_tensorflow_dynamic_graph_accepts_empty_sequences():
    import tensorflow as tf
    module = _module("tensorflow")
    encoder, relation = module.ContextualHistoryEncoder(4, 2, 4), module.RelationHistoryAttention(4, 2)
    tokens, masks = tf.TensorSpec([None, None, 4], tf.float32), tf.TensorSpec([None, None], tf.bool)

    @tf.function(input_signature=[tokens, tokens, tokens, tf.TensorSpec([None, None, 2], tf.float32), masks, masks, masks])
    def run(user, history, target, context, um, hm, tm):
        history, hm = encoder(history, context, hm)
        return relation(user, history, target, um, hm, tm, return_attention=True)

    for batch, lengths in [(2, (2, 3, 4)), (1, (0, 0, 0)), (3, (2, 0, 1))]:
        inputs = [tf.ones([batch, length, 4]) for length in lengths]
        mask_values = [tf.ones([batch, length], tf.bool) for length in lengths]
        result = run(*inputs, tf.ones([batch, lengths[1], 2]), *mask_values)
        assert result["target_history"].shape == (batch, lengths[2], 4)
        assert np.isfinite(result["target_history"].numpy()).all()


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_extracted_source_executes_without_package_dependencies(framework):
    module = _module(framework)
    namespace = {"__name__": "standalone_contextual_history"}
    exec(compile(Path(module.__file__).read_text(), "standalone_contextual_history.py", "exec"), namespace)
    inputs = _inputs(framework)
    result = namespace["RelationHistoryAttention"](4, 2)(*inputs)
    assert tuple(result["target_history"].shape) == (2, 3, 4)
    assert np.isfinite(_array(result["target_history"])).all()
