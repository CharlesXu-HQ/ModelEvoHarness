"""Analytic contracts for initial-ranking progressive interaction blocks."""

import importlib
from pathlib import Path

import numpy as np
import pytest


def _class(framework):
    pytest.importorskip("torch" if framework == "pytorch" else "tensorflow")
    path = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models" / framework / "progressive_interaction.py"
    assert path.is_file(), f"Missing standalone component: {path}"
    return getattr(importlib.import_module(
        f"model_evo_harness.models.{framework}.progressive_interaction"), "ProgressiveInteraction")


def _tensor(framework, value):
    if framework == "pytorch":
        import torch
        return torch.tensor(value)
    import tensorflow as tf
    return tf.constant(value)


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value.numpy()


def _assign(parameter, value):
    if hasattr(parameter, "assign"):
        parameter.assign(value)
    else:
        import torch
        with torch.no_grad():
            parameter.copy_(torch.tensor(value))


def _linear(layer, kernel, bias=None):
    _assign(layer.kernel if hasattr(layer, "kernel") else layer.weight,
            kernel if hasattr(layer, "kernel") else kernel.T)
    if bias is not None:
        _assign(layer.bias, bias)


def _controlled_weights(model, seed=21):
    rng = np.random.default_rng(seed)
    for block in model.blocks:
        for name in ("query", "key", "value", "projection", "ffn_in", "ffn_out"):
            layer = getattr(block, name)
            shape = tuple(layer.kernel.shape) if hasattr(layer, "kernel") else tuple(reversed(layer.weight.shape))
            _linear(layer, rng.normal(0, .3, shape).astype("float32"),
                    None if layer.bias is None else rng.normal(0, .1, tuple(layer.bias.shape)).astype("float32"))
        for name in ("attention_norm", "ffn_norm"):
            norm = getattr(block, name)
            _assign(norm.gamma if hasattr(norm, "gamma") else norm.weight,
                    np.ones(model.dim, "float32"))
            _assign(norm.beta if hasattr(norm, "beta") else norm.bias,
                    np.zeros(model.dim, "float32"))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_masks_follow_initial_descending_affinity_and_row_major_ties(framework):
    model = _class(framework)(3, 2, [0, 1, 3, 6, 9])
    x = _tensor(framework, np.array([[[3., 0.], [0., 2.], [1., 0.]]], "float32"))
    masks = _array(model.interaction_masks(x))
    # Dots: [[9,0,3],[0,4,0],[3,0,1]]. The equal 3s choose (0,2) first.
    expected = np.array([
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 0, 1], [0, 1, 0], [0, 0, 0]],
        [[1, 1, 1], [0, 1, 0], [1, 0, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    ], bool)
    np.testing.assert_array_equal(masks, expected[None])
    assert np.all(masks[:, :-1] <= masks[:, 1:])
    zeros = _tensor(framework, np.zeros((1, 3, 2), "float32"))
    np.testing.assert_array_equal(_array(model.interaction_masks(zeros))[0, 2],
                                  [[1, 1, 1], [0, 0, 0], [0, 0, 0]])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_forward_reuses_initial_masks_even_when_layer_states_change_ranking(framework):
    model = _class(framework)(3, 3, [3, 3], hidden_dim=4)
    x = _tensor(framework, np.array([[[4., 0., 0.], [0., 3., 0.], [1., 0., 0.]]], "float32"))
    model(x)
    for block in model.blocks:
        _linear(block.projection, np.zeros((3, 3), "float32"))
        _linear(block.ffn_out, np.zeros((4, 3), "float32"), np.zeros(3, "float32"))
        _assign(block.ffn_norm.gamma if hasattr(block.ffn_norm, "gamma") else block.ffn_norm.weight,
                np.array([4., 1., 1.], "float32"))
    _, masks = model(x, return_masks=True)
    expected = [[[1, 0, 1], [0, 1, 0], [0, 0, 0]]]
    np.testing.assert_array_equal(_array(masks)[:, 0], expected)
    np.testing.assert_array_equal(_array(masks)[:, 1], expected)
    intermediate = model.blocks[0](x, masks[:, 0])
    assert not np.array_equal(_array(model.interaction_masks(intermediate))[:, 1], expected)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_masked_nan_fields_and_empty_rows_have_finite_zero_padding(framework):
    model = _class(framework)(3, 3, [0, 1, 9], hidden_dim=4)
    values = np.array([[[3., 0., 1.], [np.nan, np.inf, -np.inf], [1., 0., 2.]],
                       [[np.nan] * 3] * 3], "float32")
    valid = _tensor(framework, np.array([[1, 0, 1], [0, 0, 0]], bool))
    output, masks = model(_tensor(framework, values), valid, return_masks=True)
    assert np.isfinite(_array(output)).all()
    np.testing.assert_array_equal(_array(output)[0, 1], np.zeros(3))
    np.testing.assert_array_equal(_array(output)[1], np.zeros((3, 3)))
    np.testing.assert_array_equal(_array(masks).sum((-2, -1)), [[0, 1, 4], [0, 0, 0]])
    clean = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
    np.testing.assert_allclose(_array(output), _array(model(_tensor(framework, clean), valid)), atol=1e-6)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_sparse_attention_masks_before_softmax_and_empty_rows_have_zero_context(framework):
    model = _class(framework)(3, 3, [1], hidden_dim=4)
    values = np.array([[[3., 0., 0.], [0., 2., 0.], [1., 0., 1.]]], "float32")
    x = _tensor(framework, values)
    model(x)
    block = model.blocks[0]
    for name in ("query", "key"):
        _linear(getattr(block, name), np.zeros((3, 3), "float32"))
    for name in ("value", "projection"):
        _linear(getattr(block, name), np.eye(3, dtype="float32"))
    _linear(block.ffn_out, np.zeros((4, 3), "float32"), np.zeros(3, "float32"))
    residual = values.copy()
    residual[:, 0] *= 2  # Only edge (0,0) is open; other rows get zero context.
    def norm(v):
        return (v - v.mean(-1, keepdims=True)) / np.sqrt(v.var(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(_array(model(x)), norm(norm(residual)), atol=2e-6)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_current_qk_values_change_output_without_changing_initial_ranking(framework):
    model = _class(framework)(3, 4, [9], hidden_dim=5)
    x = _tensor(framework, np.random.default_rng(3).normal(size=(1, 3, 4)).astype("float32"))
    model(x)
    _controlled_weights(model)
    before, masks = model(x, return_masks=True)
    _linear(model.blocks[0].query, np.eye(4, dtype="float32") * 6)
    after, new_masks = model(x, return_masks=True)
    assert not np.allclose(_array(before), _array(after), atol=1e-5)
    np.testing.assert_array_equal(_array(masks), _array(new_masks))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_full_budget_matches_dense_attention_and_schedules_are_parameter_matched(framework):
    cls = _class(framework)
    x = _tensor(framework, np.random.default_rng(7).normal(size=(2, 3, 4)).astype("float32"))
    counts = []
    models = []
    for schedule in ([1, 9], [1, 1], [9, 9]):
        model = cls(3, 4, schedule, hidden_dim=5)
        model(x)
        _controlled_weights(model)
        variables = model.parameters() if framework == "pytorch" else model.trainable_variables
        counts.append(sum(int(np.prod(tuple(p.shape))) for p in variables))
        models.append(model)
    # Per layer: Q/K/V/O 4D² + two FFN matrices 2DH + H+D biases + two LN 4D.
    assert counts == [258, 258, 258]
    dense = models[-1]
    states = x
    for block in dense.blocks:
        if framework == "pytorch":
            import torch
            attention = torch.nn.functional.scaled_dot_product_attention(
                block.query(states), block.key(states), block.value(states))
            states = block.attention_norm(states + block.projection(attention))
            states = block.ffn_norm(states + block.ffn_out(torch.nn.functional.gelu(block.ffn_in(states))))
        else:
            import tensorflow as tf
            logits = tf.linalg.matmul(block.query(states), block.key(states), transpose_b=True) / 2
            attention = tf.linalg.matmul(tf.nn.softmax(logits), block.value(states))
            states = block.attention_norm(states + block.projection(attention))
            states = block.ffn_norm(states + block.ffn_out(tf.nn.gelu(block.ffn_in(states))))
    np.testing.assert_allclose(_array(dense(x)), _array(states), atol=2e-6)
    assert not np.allclose(_array(models[0](x)), _array(models[1](x)))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
@pytest.mark.parametrize("budgets", [[], [-1], [10], [3, 2], [1.5], [True]])
def test_rejects_invalid_global_edge_budgets(framework, budgets):
    with pytest.raises(ValueError, match="budget"):
        _class(framework)(3, 4, budgets)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_rejects_invalid_shapes_and_nonboolean_validity(framework):
    model = _class(framework)(3, 4, [9])
    errors = (ValueError, TypeError)
    if framework == "tensorflow":
        import tensorflow as tf
        errors += (tf.errors.InvalidArgumentError,)
    with pytest.raises(errors):
        model(_tensor(framework, np.ones((2, 4), "float32")))
    with pytest.raises(errors):
        model(_tensor(framework, np.ones((1, 3, 4), "float32")),
              _tensor(framework, np.ones((1, 3), "float32")))
    with pytest.raises(ValueError):
        _class(framework)(0, 4, [0])


def test_framework_output_input_gradient_and_parameter_gradient_agreement():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("tensorflow")
    values = np.random.default_rng(9).normal(size=(2, 3, 4)).astype("float32")
    valid = np.array([[1, 0, 1], [1, 1, 1]], bool)
    values[~valid] = np.nan
    pt = _class("pytorch")(3, 4, [3, 9], hidden_dim=5)
    kt = _class("tensorflow")(3, 4, [3, 9], hidden_dim=5)
    kt(tf.constant(values), tf.constant(valid))
    _controlled_weights(pt)
    _controlled_weights(kt)
    x = torch.tensor(values, requires_grad=True)
    y = tf.Variable(values)
    coefficients = np.random.default_rng(11).normal(size=values.shape).astype("float32")
    out_pt, mask_pt = pt(x, torch.tensor(valid), return_masks=True)
    (out_pt * torch.tensor(coefficients)).sum().backward()
    with tf.GradientTape() as tape:
        out_tf, mask_tf = kt(y, tf.constant(valid), return_masks=True)
        loss = tf.reduce_sum(out_tf * coefficients)
    grads = tape.gradient(loss, [y] + kt.trainable_variables)
    np.testing.assert_array_equal(_array(mask_pt), _array(mask_tf))
    np.testing.assert_allclose(_array(out_pt), _array(out_tf), atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(_array(x.grad), _array(grads[0]), atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(_array(x.grad)[~valid], np.zeros((1, 4)))
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in pt.parameters())
    assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in grads)
    # Current Q/K receive useful gradients; the hard initial ranking has no trainable weights.
    for name in ("query", "key"):
        grad_pt = getattr(pt.blocks[-1], name).weight.grad
        parameter_tf = getattr(kt.blocks[-1], name).kernel
        grad_tf = grads[1 + next(i for i, v in enumerate(kt.trainable_variables) if v is parameter_tf)]
        assert np.linalg.norm(_array(grad_pt)) > 0
        np.testing.assert_allclose(_array(grad_pt).T, _array(grad_tf), atol=1e-5, rtol=1e-5)


def test_tensorflow_graph_supports_dynamic_batches_and_all_masked_gradients():
    tf = pytest.importorskip("tensorflow")
    model = _class("tensorflow")(3, 4, [0, 2, 9], hidden_dim=5)

    @tf.function(input_signature=[tf.TensorSpec([None, 3, 4], tf.float32),
                                  tf.TensorSpec([None, 3], tf.bool)], autograph=False)
    def run(x, valid):
        with tf.GradientTape() as tape:
            tape.watch(x)
            out, masks = model(x, valid, return_masks=True)
            loss = tf.reduce_sum(out * tf.range(4, dtype=tf.float32))
        return out, masks, tape.gradient(loss, x)

    for batch in (1, 2):
        values = tf.fill([batch, 3, 4], float("nan"))
        out, masks, grads = run(values, tf.zeros([batch, 3], tf.bool))
        np.testing.assert_array_equal(_array(out), np.zeros((batch, 3, 4)))
        np.testing.assert_array_equal(_array(masks), np.zeros((batch, 3, 3, 3), bool))
        np.testing.assert_array_equal(_array(grads), np.zeros((batch, 3, 4)))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_source_extracts_without_project_imports(framework):
    cls = _class(framework)
    module = importlib.import_module(cls.__module__)
    namespace = {"__name__": "extracted_progressive_interaction"}
    exec(compile(Path(module.__file__).read_text(), "extracted_progressive_interaction.py", "exec"), namespace)
    model = namespace["ProgressiveInteraction"](2, 3, [1, 4])
    output = model(_tensor(framework, np.ones((1, 2, 3), "float32")))
    assert tuple(output.shape) == (1, 2, 3)
    assert np.isfinite(_array(output)).all()
