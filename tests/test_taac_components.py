"""Equation, dependency and masking contracts for the first TAAC components."""

import importlib
import math
from pathlib import Path

import numpy as np
import pytest


def _class(framework, name):
    module = "paper_interactions" if name in {
        "RankMixerBlock", "ExplicitInteractionHighway"} else "paper_unified_sequence"
    path = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models" / framework / f"{module}.py"
    assert path.is_file(), f"Missing standalone component module: {path}"
    return getattr(importlib.import_module(f"model_evo_harness.models.{framework}.{module}"), name)


def _tensor(framework, value):
    if framework == "pytorch":
        import torch
        return torch.tensor(value)
    import tensorflow as tf
    return tf.constant(value)


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value.numpy()


def _assign(parameter, value):
    value = np.asarray(value, np.float32)
    if hasattr(parameter, "assign"):
        parameter.assign(value)
    else:
        import torch
        with torch.no_grad():
            parameter.copy_(torch.tensor(value))


def _linear(layer, kernel, bias):
    if hasattr(layer, "kernel"):
        _assign(layer.kernel, kernel)
    else:
        _assign(layer.weight, np.asarray(kernel).T)
    _assign(layer.bias, bias)


def _ln(x):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_rankmixer_rewiring_post_norm_and_independent_ffn(framework):
    cls = _class(framework, "RankMixerBlock")
    model = cls(2, 4, 3)
    x = _tensor(framework, np.arange(1, 9, dtype=np.float32).reshape(1, 2, 4))
    model(x)  # Build Keras variables before assigning controlled weights.
    np.testing.assert_array_equal(_array(model.mix_tokens(x)), [[[1, 2, 5, 6], [3, 4, 7, 8]]])
    np.testing.assert_array_equal(_array(model.mix_tokens(model.mix_tokens(x))), _array(x))
    for name in ("w1", "b1", "w2", "b2"):
        p = getattr(model, name)
        _assign(p, np.zeros(tuple(p.shape)))
    first = _array(model(x))
    np.testing.assert_allclose(first, [[[-1.264905, -.632452, .632452, 1.264905]] * 2], atol=2e-5)
    # A token-specific FFN bias must not change the other token's FFN output.
    _assign(model.b2, [[1, 0, 0, 0], [0, 0, 0, 0]])
    second = _array(model(x))
    assert not np.allclose(first[:, 0], second[:, 0])
    np.testing.assert_array_equal(first[:, 1], second[:, 1])
    with pytest.raises(ValueError, match="divisible"):
        cls(3, 4, 8)
    error = ValueError
    if framework == "tensorflow":
        import tensorflow as tf
        error = tf.errors.InvalidArgumentError
    with pytest.raises(error, match="[Ss]hape"):
        model(_tensor(framework, np.zeros((1, 3, 4), np.float32)))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_condition_enters_gate_only_and_padding_is_sanitized(framework):
    model = _class(framework, "ConditionalSwiGLU")(2, 1, 2, 2)
    x = _tensor(framework, np.array([[[1., 2.], [np.nan, np.inf]]], np.float32))
    c = _tensor(framework, np.array([[math.log(3)]], np.float32))
    mask = _tensor(framework, np.array([[True, False]]))
    model(x, c, mask)
    _linear(model.value, np.eye(2), np.zeros(2))
    _linear(model.gate, [[0, 0], [0, 0], [1, 1]], np.zeros(2))
    _linear(model.output_projection, np.eye(2), np.zeros(2))
    np.testing.assert_allclose(_array(model(x, c, mask)),
                               [[[.75 * math.log(3), 1.5 * math.log(3)], [0, 0]]], atol=1e-6)
    _linear(model.value, np.zeros((2, 2)), np.zeros(2))
    np.testing.assert_array_equal(_array(model(x, c, mask)), np.zeros((1, 2, 2)))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_multichannel_pool_matches_sigmoid_not_softmax(framework):
    model = _class(framework, "MultiChannelSequencePool")(3, 1, 2, 1)
    state = _tensor(framework, np.zeros((1, 1, 3), np.float32))
    history = _tensor(framework, np.array([[[1., 0., 0.], [0., 2., 0.]]], np.float32))
    mask = _tensor(framework, np.ones((1, 2), bool))
    model(state, [history], [mask])
    _linear(model.state_projection, np.zeros((3, 3)), np.zeros(3))
    scorer = model.scorers[0]
    layers = scorer.layers if hasattr(scorer, "layers") else scorer
    first, last = layers[0], layers[-1]
    kernel = np.zeros((6, 3)); kernel[0, 0] = 1
    _linear(first, kernel, np.zeros(3))
    _linear(last, [[1, -1], [0, 0], [0, 0]], np.zeros(2))
    _linear(model.pool_projections[0], np.eye(3), np.zeros(3))
    gelu_one = .5 * (1 + math.erf(1 / math.sqrt(2)))
    a = 1 / (1 + math.exp(-gelu_one))
    # Each channel has independent sigmoid gates: (a,.5) and (1-a,.5).
    pooled = np.array([[[a, 1, 0], [1-a, 1, 0]]])
    expected = _ln(pooled / np.linalg.norm(pooled, axis=-1, keepdims=True))
    np.testing.assert_allclose(_array(model(state, [history], [mask])), expected, atol=2e-6)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_pool_empty_domains_and_masked_nonfinite_values(framework):
    model = _class(framework, "MultiChannelSequencePool")(3, 2, 2, 2)
    rng = np.random.default_rng(10)
    state = _tensor(framework, rng.normal(size=(2, 2, 3)).astype("float32"))
    values = rng.normal(size=(2, 4, 3)).astype("float32")
    valid = np.array([[True, False, True, False], [False] * 4])
    masks = [_tensor(framework, valid), _tensor(framework, np.zeros((2, 0), bool))]
    empty = _tensor(framework, np.zeros((2, 0, 3), np.float32))
    model(state, [_tensor(framework, values), empty], masks)
    # Learned biases must not turn absent domains into fabricated evidence.
    for projection in model.pool_projections:
        _assign(projection.bias, [2, -1, 3])
    _assign(model.out_norm.beta if hasattr(model.out_norm, "beta") else model.out_norm.bias, [3, 2, 1])
    first = _array(model(state, [_tensor(framework, values), empty], masks))
    values[~valid] = np.nan
    second = _array(model(state, [_tensor(framework, values), empty], masks))
    np.testing.assert_allclose(first, second)
    np.testing.assert_array_equal(second[1], np.zeros((4, 3)))
    np.testing.assert_array_equal(second[:, 2:], np.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="domain"):
        model(state, [empty], masks)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_highway_raw_domain_dots_and_cross_layer_readout(framework):
    cls = _class(framework, "ExplicitInteractionHighway")
    query = np.array([[[1., 2., 3.], [4., 5., 6.]]], np.float32)
    reads = np.array([[[[2., 0., 1.], [1., 1., 0.]], [[np.nan]*3, [np.inf]*3]]], np.float32)
    dots = cls.domain_interactions(_tensor(framework, query), _tensor(framework, reads),
                                   _tensor(framework, np.array([[True, False]])))
    np.testing.assert_array_equal(_array(dots), [[[5, 9], [0, 0]]])
    model = cls(2, 3, 1, 2, num_domains=1)
    users = np.array([[[[-1., 0., 1.]], [[1., 0., -1.]]]], np.float32)
    items = np.array([[[[1., 0., -1.], [0., 1., -1.]],
                       [[1., 0., -1.], [0., 1., -1.]]]], np.float32)
    block = np.array([[[[1., 2.]], [[4., 8.]]]], np.float32)
    output = _array(model(_tensor(framework, users), _tensor(framework, items), _tensor(framework, block)))
    # Norms across D on each side; flatten layer, user, item; normalize each bundle separately.
    ui = np.array([[-3., -1.5, 3., 1.5]]) / (1 + 1.5e-5)
    expected = np.concatenate((_ln(block.reshape(1, 4)), _ln(ui)), -1)
    np.testing.assert_allclose(output, expected, atol=2e-6)
    # Field-only Highway needs no history or manufactured domain tensor.
    plain = cls(2, 3, 1, 2)
    np.testing.assert_allclose(_array(plain(_tensor(framework, users), _tensor(framework, items))), expected[:, 4:], atol=2e-6)
    with pytest.raises(ValueError, match="domain"):
        model(_tensor(framework, users), _tensor(framework, items))
    with pytest.raises(ValueError, match="at least two"):
        cls(1, 3, 1, 1)


def _training_gradients(framework):
    rng = np.random.default_rng(12)
    history = rng.normal(size=(2, 4, 4)).astype("float32")
    state = rng.normal(size=(2, 2, 4)).astype("float32")
    valid = np.array([[True, True, False, False], [True, False, False, False]])
    gate = _class(framework, "ConditionalSwiGLU")(4, 4, 7, 4)
    pool = _class(framework, "MultiChannelSequencePool")(4, 2, 2, 1)
    mixer = _class(framework, "RankMixerBlock")(4, 4, 7)
    highway = _class(framework, "ExplicitInteractionHighway")(2, 4, 2, 2, num_domains=1)

    def compute(x, fields):
        h = gate(x, fields[:, 1], _tensor(framework, valid))
        pooled = pool(fields, [h], [_tensor(framework, valid)])
        if framework == "pytorch":
            import torch as ops
            mixed = mixer(ops.cat((fields, pooled), 1))
            users = ops.stack((fields, mixed[:, :2]), 1)
            items = ops.stack((pooled, mixed[:, 2:]), 1)
            dots = ops.stack(((fields * pooled).sum(-1), (mixed[:, :2] * mixed[:, 2:]).sum(-1)), 1)[:, :, None]
            features = highway(users, items, dots)
            return (features * ops.arange(1, 13, dtype=features.dtype)).sum()
        import tensorflow as tf
        mixed = mixer(tf.concat((fields, pooled), 1))
        users = tf.stack((fields, mixed[:, :2]), 1)
        items = tf.stack((pooled, mixed[:, 2:]), 1)
        dots = tf.stack((tf.reduce_sum(fields * pooled, -1), tf.reduce_sum(mixed[:, :2] * mixed[:, 2:], -1)), 1)[:, :, None]
        features = highway(users, items, dots)
        return tf.reduce_sum(features * tf.range(1, 13, dtype=features.dtype))

    if framework == "pytorch":
        import torch
        x = torch.tensor(history, requires_grad=True)
        fields = torch.tensor(state, requires_grad=True)
        compute(x, fields).backward()
        gradients = [x.grad, fields.grad] + [p.grad for model in (gate, pool, mixer, highway) for p in model.parameters()]
    else:
        import tensorflow as tf
        x, fields = tf.Variable(history), tf.Variable(state)
        with tf.GradientTape() as tape:
            loss = compute(x, fields)
        gradients = tape.gradient(loss, [x, fields] + [p for model in (gate, pool, mixer, highway) for p in model.trainable_variables])
    assert all(g is not None and np.isfinite(_array(g)).all() for g in gradients)
    np.testing.assert_array_equal(_array(gradients[0])[~valid], 0)
    assert np.linalg.norm(_array(gradients[0])[valid]) > 1e-6
    assert np.linalg.norm(_array(gradients[1])) > 1e-6
    return gradients


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_all_four_components_train_and_masked_positions_have_zero_gradient(framework):
    _training_gradients(framework)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_all_four_components_gpu_forward_backward(framework):
    if framework == "pytorch":
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU unavailable")
        with torch.device("cuda:0"):
            gradients = _training_gradients(framework)
        assert gradients[0].is_cuda
    else:
        import tensorflow as tf
        if not tf.config.list_physical_devices("GPU"):
            pytest.skip("TensorFlow GPU unavailable")
        with tf.device("/GPU:0"):
            gradients = _training_gradients(framework)
        assert "GPU:0" in gradients[0].device.upper()


def test_tensorflow_graph_accepts_dynamic_batch_and_history_lengths():
    import tensorflow as tf
    gate = _class("tensorflow", "ConditionalSwiGLU")(4, 4, 6, 4)
    pool = _class("tensorflow", "MultiChannelSequencePool")(4, 2, 2, 1)
    mixer = _class("tensorflow", "RankMixerBlock")(4, 4, 6)
    highway = _class("tensorflow", "ExplicitInteractionHighway")(2, 4, 2, 2)

    @tf.function(input_signature=[tf.TensorSpec((None, None, 4), tf.float32),
                                  tf.TensorSpec((None, 2, 4), tf.float32),
                                  tf.TensorSpec((None, None), tf.bool)])
    def run(history, fields, mask):
        states = gate(history, fields[:, 1], mask)
        pooled = pool(fields, [states], [mask])
        mixed = mixer(tf.concat((fields, pooled), 1))
        return highway(tf.stack((fields, mixed[:, :2]), 1), tf.stack((pooled, mixed[:, 2:]), 1))

    for batch, length in ((2, 3), (1, 7), (3, 0)):
        result = run(tf.ones((batch, length, 4)), tf.ones((batch, 2, 4)), tf.ones((batch, length), tf.bool))
        assert tuple(result.shape) == (batch, 8)
        assert np.isfinite(result.numpy()).all()


@pytest.mark.parametrize("name", ["RankMixerBlock", "ConditionalSwiGLU",
                                  "MultiChannelSequencePool", "ExplicitInteractionHighway"])
def test_same_weight_framework_outputs_and_input_gradients(name):
    import torch
    import tensorflow as tf
    torch.manual_seed(23)
    rng = np.random.default_rng(23)

    def random(shape):
        return rng.normal(size=shape).astype("float32")

    constructors = {"RankMixerBlock": (2, 4, 6), "ConditionalSwiGLU": (4, 3, 6, 4),
                    "MultiChannelSequencePool": (4, 2, 2, 2),
                    "ExplicitInteractionHighway": (2, 4, 2, 2, 1)}
    data = {
        "RankMixerBlock": [random((2, 2, 4))],
        "ConditionalSwiGLU": [random((2, 3, 4)), random((2, 3))],
        "MultiChannelSequencePool": [random((2, 2, 4)), random((2, 3, 4)), random((2, 3, 4))],
        "ExplicitInteractionHighway": [random((2, 2, 2, 4)), random((2, 2, 2, 4)), random((2, 2, 1, 2))],
    }[name]
    mask = np.array([[True, False, True], [False, True, True]])
    pt = _class("pytorch", name)(*constructors[name])
    tfm = _class("tensorflow", name)(*constructors[name])

    def call(model, tensors, framework):
        if name == "ConditionalSwiGLU":
            return model(*tensors, _tensor(framework, mask))
        if name == "MultiChannelSequencePool":
            return model(tensors[0], tensors[1:], [_tensor(framework, mask)] * 2)
        return model(*tensors)

    call(tfm, [tf.constant(x) for x in data], "tensorflow")

    def linear(a, b):
        _linear(b, a.weight.detach().numpy().T, a.bias.detach().numpy())

    def norm(a, b):
        _assign(b.gamma, a.weight.detach().numpy())
        _assign(b.beta, a.bias.detach().numpy())

    if name == "RankMixerBlock":
        for attr in ("w1", "b1", "w2", "b2"):
            _assign(getattr(tfm, attr), _array(getattr(pt, attr)))
        norm(pt.mix_norm, tfm.mix_norm)
        norm(pt.ffn_norm, tfm.ffn_norm)
    elif name == "ConditionalSwiGLU":
        for attr in ("value", "gate", "output_projection"):
            linear(getattr(pt, attr), getattr(tfm, attr))
    elif name == "MultiChannelSequencePool":
        linear(pt.state_projection, tfm.state_projection)
        norm(pt.out_norm, tfm.out_norm)
        for index in range(2):
            linear(pt.scorers[index][0], tfm.scorers[index].layers[0])
            linear(pt.scorers[index][-1], tfm.scorers[index].layers[-1])
            linear(pt.pool_projections[index], tfm.pool_projections[index])
    else:
        for attr in ("user_norm", "item_norm", "domain_norm", "cross_norm"):
            norm(getattr(pt, attr), getattr(tfm, attr))

    variables = [torch.tensor(x, requires_grad=True) for x in data]
    output = call(pt, variables, "pytorch")
    weights = rng.normal(size=tuple(output.shape)).astype("float32")
    (output * torch.tensor(weights)).sum().backward()
    tf_vars = [tf.Variable(x) for x in data]
    with tf.GradientTape() as tape:
        tf_output = call(tfm, tf_vars, "tensorflow")
        loss = tf.reduce_sum(tf_output * weights)
    tf_grads = tape.gradient(loss, tf_vars)
    np.testing.assert_allclose(_array(output), _array(tf_output), rtol=2e-5, atol=3e-6)
    for value, gradient in zip(variables, tf_grads):
        assert gradient is not None and value.grad is not None
        np.testing.assert_allclose(_array(value.grad), _array(gradient), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_half_precision_pool_empty_and_zero_vectors_have_finite_backward(framework):
    cls = _class(framework, "MultiChannelSequencePool")
    mask = np.array([[False, False], [True, True]])
    if framework == "pytorch":
        import torch
        model = cls(4, 1, 2, 1).half()
        history = torch.zeros((2, 2, 4), dtype=torch.float16, requires_grad=True)
        output = model(torch.ones((2, 1, 4), dtype=torch.float16), [history], [torch.tensor(mask)])
        (output * torch.tensor([1., 2., 3., 4.], dtype=torch.float16)).sum().backward()
        gradients = [history.grad] + [p.grad for p in model.parameters()]
    else:
        import tensorflow as tf
        policy = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            model = cls(4, 1, 2, 1)
            history = tf.Variable(tf.zeros((2, 2, 4), dtype=tf.float16))
            with tf.GradientTape() as tape:
                output = model(tf.ones((2, 1, 4), tf.float16), [history], [tf.constant(mask)])
                loss = tf.reduce_sum(output * tf.constant([1., 2., 3., 4.], tf.float16))
            gradients = tape.gradient(loss, [history] + model.trainable_variables)
        finally:
            tf.keras.mixed_precision.set_global_policy(policy)
    assert np.isfinite(_array(output)).all()
    np.testing.assert_array_equal(_array(output)[0], 0)
    assert all(g is not None and np.isfinite(_array(g)).all() for g in gradients)
    np.testing.assert_array_equal(_array(gradients[0]), 0)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_extracted_modules_run_without_repository_imports(framework):
    from model_evo_harness import load_catalog, read_references
    references = read_references(load_catalog(), {"framework": framework, "method_ids": [
        "rankmixer", "unidot_conditioned_gate", "unidot_multichannel_pool", "unidot_fm_highway"]})
    classes = {}
    for path, record in references["files"].items():
        namespace = {"__name__": "extracted_taac"}
        exec(compile(record["content"], path, "exec"), namespace)
        classes.update(namespace)
    tensor = lambda x: _tensor(framework, np.asarray(x, np.float32))
    fields = tensor(np.ones((1, 2, 4)))
    mask = _tensor(framework, np.ones((1, 3), bool))
    history = classes["ConditionalSwiGLU"](4, 4, 6, 4)(tensor(np.ones((1, 3, 4))), fields[:, 1], mask)
    pooled = classes["MultiChannelSequencePool"](4, 2, 2, 1)(fields, [history], [mask])
    mixed = classes["RankMixerBlock"](2, 4, 6)(pooled)
    terms = classes["ExplicitInteractionHighway"](1, 4, 2, 2)(fields[:, None], mixed[:, None])
    assert tuple(terms.shape) == (1, 4)
    assert np.isfinite(_array(terms)).all()
