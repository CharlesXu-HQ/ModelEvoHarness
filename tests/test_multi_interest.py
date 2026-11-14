"""Independent numeric and gradient contracts for native ComiRec extractors."""
import importlib
import math
from pathlib import Path

import numpy as np
import pytest


FRAMEWORKS = ["pytorch", "tensorflow"]
SPECS = [("ComiRecSA", {"hidden_dim": 3, "max_length": 5}),
         ("ComiRecDR", {"max_length": 5})]


def module(framework):
    path = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models" / framework / "multi_interest.py"
    assert path.is_file(), f"Missing multi-interest implementation: {path}"
    return importlib.import_module(f"model_evo_harness.models.{framework}.multi_interest")


def tensor(framework, value):
    value = np.asarray(value)
    if value.dtype != bool:
        value = value.astype(np.float32)
    if framework == "pytorch":
        import torch
        return torch.tensor(value)
    import tensorflow as tf
    return tf.constant(value)


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value.numpy()


def assign(parameter, value):
    if hasattr(parameter, "assign"):
        parameter.assign(value)
    else:
        import torch
        with torch.no_grad():
            parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype))


def linear(layer, kernel):
    if hasattr(layer, "kernel"):
        assign(layer.kernel, kernel)
    else:
        assign(layer.weight, np.asarray(kernel).T)


def fixture(framework, length):
    valid = np.zeros((2, length), bool)
    valid[0, :min(2, length)] = True
    history = np.where(valid[..., None], np.random.default_rng(19).normal(size=(2, length, 2)), np.nan)
    return tensor(framework, history), tensor(framework, valid)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("positions", [False, True])
def test_sa_normalizes_over_time_and_positions_change_only_scores(framework, positions):
    model = module(framework).ComiRecSA(2, 2, hidden_dim=1, max_length=3 if positions else None)
    t = lambda x: tensor(framework, x)
    x = 0.0 if positions else math.atanh(.5)
    history, valid = t([[[x, 2], [-x, 4], [np.nan, np.inf]]]), t([[True, True, False]])
    model(history, valid)
    linear(model.hidden, [[1], [0]])
    linear(model.score, [[1, -1]])
    if positions:
        assign(model.position, [[math.atanh(.5), 0], [0, 0], [20, 20]])
    result = model(history, valid)
    p = 1 / (1 + math.exp(-(.5 if positions else 1)))
    np.testing.assert_allclose(array(result["assignments"]), [[[p, 1-p, 0], [1-p, p, 0]]], atol=1e-6)
    np.testing.assert_allclose(array(result["interests"]), [[[x*(2*p-1), 4-2*p], [x*(1-2*p), 2+2*p]]], atol=1e-6)
    np.testing.assert_array_equal(array(result["interest_mask"]), [[True, True]])


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_dr_uses_each_position_interest_matrix_and_capsule_squash(framework):
    model = module(framework).ComiRecDR(2, 2, max_length=3, routing_iterations=1)
    projection = np.zeros((3, 2, 2, 2), np.float32)
    projection[:2] = [[[[1, 2], [0, 1]], [[0, 1], [2, 0]]],
                      [[[1, 0], [1, 2]], [[0, 2], [1, 1]]]]
    assign(model.projection, projection)
    result = model(tensor(framework, [[[1, 2], [3, 1], [np.nan, np.nan]]]),
                   tensor(framework, [[True, True, False]]))
    np.testing.assert_allclose(array(result["assignments"]), [[[.5, .5, 0], [.5, .5, 0]]])
    expected = [[[2.5*math.sqrt(15.25)/16.25, 3*math.sqrt(15.25)/16.25],
                 [2.5*math.sqrt(22.25)/23.25, 4*math.sqrt(22.25)/23.25]]]
    np.testing.assert_allclose(array(result["interests"]), expected, atol=1e-6)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_dr_agreement_routes_each_event_over_interests(framework):
    model = module(framework).ComiRecDR(1, 2, max_length=2, routing_iterations=2)
    assign(model.projection, [[[[1]], [[2]]], [[[1]], [[.5]]]])
    result = model(tensor(framework, [[[1], [2]]]), tensor(framework, [[True, True]]))
    p = 1 / (1 + math.exp(-9/13))
    np.testing.assert_allclose(array(result["assignments"]), [[[1-p, p], [p, 1-p]]], atol=1e-6)
    squashed = (1+p)**2 / (1+(1+p)**2)
    np.testing.assert_allclose(array(result["interests"]), [[[squashed], [squashed]]], atol=1e-6)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("stop", [False, True])
def test_dr_stops_only_nonfinal_routing_votes_with_known_derivative(framework, stop):
    model = module(framework).ComiRecDR(1, 2, max_length=1, routing_iterations=2, stop_gradient_routing=stop)
    assign(model.projection, [[[[1]], [[2]]]])
    history, valid = tensor(framework, [[[1]]]), tensor(framework, [[True]])
    if framework == "pytorch":
        history.requires_grad_(True)
        result = model(history, valid)
        result["interests"].sum().backward()
        gradient = history.grad
        projection_gradient = model.projection.grad
        assert result["assignments"].requires_grad is (not stop)
    else:
        import tensorflow as tf
        history = tf.Variable(history)
        with tf.GradientTape(persistent=True) as tape:
            result = model(history, valid)
            loss = tf.reduce_sum(result["interests"])
            coefficient = result["assignments"][0, 0, 0]
        gradient, projection_gradient = tape.gradient(loss, [history, model.projection])
        assert (tape.gradient(coefficient, history) is None) is stop
    c = 1 / (1 + math.exp(.8))
    dc = 0 if stop else -1.48*c*(1-c)
    expected = 2*c/(1+c*c)**2 * (c+dc) + 4*(1-c)/(1+4*(1-c)**2)**2 * (2*(1-c)-2*dc)
    np.testing.assert_allclose(array(gradient), [[[expected]]], atol=1e-6)
    assert projection_gradient is not None and np.isfinite(array(projection_gradient)).all()
    assert np.any(array(projection_gradient) != 0)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("name,kwargs", SPECS)
@pytest.mark.parametrize("length", [0, 3])
def test_masks_nan_padding_and_empty_history_have_zero_finite_outputs_and_gradients(framework, name, kwargs, length):
    model = getattr(module(framework), name)(2, 3, **kwargs)
    history, valid = fixture(framework, length)
    if framework == "pytorch":
        history.requires_grad_(True)
        result = model(history, valid)
        result["interests"].sum().backward()
        gradients = [history.grad] + [p.grad for p in model.parameters()]
    else:
        import tensorflow as tf
        history = tf.Variable(history)
        with tf.GradientTape() as tape:
            result = model(history, valid)
            loss = tf.reduce_sum(result["interests"])
        gradients = tape.gradient(loss, [history] + model.trainable_variables)
    assert tuple(result["interests"].shape) == (2, 3, 2)
    assert tuple(result["assignments"].shape) == (2, 3, length)
    np.testing.assert_array_equal(array(result["interest_mask"]), [[bool(length)]*3, [False]*3])
    for key in ("interests", "assignments"):
        value = array(result[key])
        assert np.isfinite(value).all()
        np.testing.assert_array_equal(value[1], 0)
        if not length:
            np.testing.assert_array_equal(value, 0)
    assert all(g is not None and np.isfinite(array(g)).all() for g in gradients)
    np.testing.assert_array_equal(array(gradients[0])[~array(valid)], 0)
    if length:
        assert np.any(array(gradients[0])[array(valid)] != 0)
        assert any(np.any(array(g) != 0) for g in gradients[1:])
        axis = -1 if name == "ComiRecSA" else 1
        expected = np.ones(3) if name == "ComiRecSA" else array(valid)[0]
        np.testing.assert_allclose(array(result["assignments"])[0].sum(axis=axis if axis == -1 else 0), expected, atol=1e-6)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_dr_zero_vectors_have_finite_zero_gradients(framework):
    model = module(framework).ComiRecDR(2, 2, max_length=2)
    history, valid = tensor(framework, np.zeros((1, 2, 2))), tensor(framework, [[True, True]])
    if framework == "pytorch":
        history.requires_grad_(True)
        result = model(history, valid)
        result["interests"].sum().backward()
        gradients = [history.grad, model.projection.grad]
    else:
        import tensorflow as tf
        history = tf.Variable(history)
        with tf.GradientTape() as tape:
            result = model(history, valid)
            loss = tf.reduce_sum(result["interests"])
        gradients = tape.gradient(loss, [history, model.projection])
    np.testing.assert_array_equal(array(result["interests"]), 0)
    for gradient in gradients:
        assert gradient is not None
        np.testing.assert_array_equal(array(gradient), 0)


@pytest.mark.parametrize("name,kwargs", SPECS)
def test_mapped_weights_match_outputs_and_gradients_across_frameworks(name, kwargs):
    import torch
    import tensorflow as tf
    torch.manual_seed(22)
    first = getattr(module("pytorch"), name)(2, 3, **kwargs)
    second = getattr(module("tensorflow"), name)(2, 3, **kwargs)
    if name == "ComiRecSA":
        linear(second.hidden, array(first.hidden.weight).T)
        linear(second.score, array(first.score.weight).T)
        assign(second.position, array(first.position))
    else:
        assign(second.projection, array(first.projection))
    for length in (0, 3, 5):
        history, valid = fixture("pytorch", length)
        history.requires_grad_(True)
        pt = first(history, valid)
        pt["interests"].sum().backward()
        tf_history, tf_valid = fixture("tensorflow", length)
        tf_history = tf.Variable(tf_history)
        with tf.GradientTape() as tape:
            other = second(tf_history, tf_valid)
            loss = tf.reduce_sum(other["interests"])
        tf_gradient = tape.gradient(loss, tf_history)
        for key in pt:
            np.testing.assert_allclose(array(pt[key]), array(other[key]), atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(array(history.grad), array(tf_gradient), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("name,kwargs", SPECS)
def test_tensorflow_dynamic_graph_accepts_zero_and_variable_lengths(name, kwargs):
    import tensorflow as tf
    model = getattr(module("tensorflow"), name)(2, 3, **kwargs)
    @tf.function(autograph=False, input_signature=[tf.TensorSpec([None, None, 2], tf.float32), tf.TensorSpec([None, None], tf.bool)])
    def run(history, valid):
        return model(history, valid)
    for length in (3, 0, 5):
        result = run(*fixture("tensorflow", length))
        assert tuple(result["interests"].shape) == (2, 3, 2)
        assert np.isfinite(array(result["interests"])).all()
        np.testing.assert_array_equal(array(result["interests"])[1], 0)
    with pytest.raises(tf.errors.InvalidArgumentError):
        run(*fixture("tensorflow", 6))


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("name,kwargs", SPECS)
def test_invalid_shapes_types_and_lengths_are_rejected(framework, name, kwargs):
    import tensorflow as tf
    model = getattr(module(framework), name)(2, 3, **kwargs)
    t = lambda x: tensor(framework, x)
    cases = [(np.zeros((1, 2)), np.ones((1, 2), bool)),
             (np.zeros((1, 2, 3)), np.ones((1, 2), bool)),
             (np.zeros((1, 2, 2)), np.ones((1, 1), bool)),
             (np.zeros((1, 6, 2)), np.ones((1, 6), bool)),
             (np.zeros((1, 2, 2)), np.ones((1, 2), np.float32))]
    for history, valid in cases:
        with pytest.raises((ValueError, TypeError, tf.errors.InvalidArgumentError)):
            model(t(history), t(valid))
    if framework == "pytorch":
        import torch
        integer_history = torch.ones((1, 2, 2), dtype=torch.int64)
    else:
        integer_history = tf.ones((1, 2, 2), tf.int64)
    with pytest.raises((ValueError, TypeError, tf.errors.InvalidArgumentError)):
        model(integer_history, t([[True, True]]))


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("name,kwargs", [
    ("ComiRecSA", dict(dim=0, num_interests=2)),
    ("ComiRecSA", dict(dim=2, num_interests=0)),
    ("ComiRecSA", dict(dim=2, num_interests=2, hidden_dim=0)),
    ("ComiRecSA", dict(dim=2, num_interests=2, max_length=0)),
    ("ComiRecSA", dict(dim=2.5, num_interests=2)),
    ("ComiRecDR", dict(dim=2, num_interests=2, max_length=0)),
    ("ComiRecDR", dict(dim=2, num_interests=2, max_length=3, routing_iterations=0)),
    ("ComiRecDR", dict(dim=2, num_interests=2, max_length=3, routing_iterations=1.5)),
    ("ComiRecDR", dict(dim=2, num_interests=2, max_length=3, stop_gradient_routing="yes")),
])
def test_invalid_configuration_is_rejected(framework, name, kwargs):
    with pytest.raises((TypeError, ValueError)):
        getattr(module(framework), name)(**kwargs)
