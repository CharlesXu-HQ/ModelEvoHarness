"""Formula, parameter-generation and gradient boundaries for conditional layers."""
import importlib
from pathlib import Path

import numpy as np
import pytest


def cls(framework, name):
    path = Path(__file__).resolve().parents[1] / 'src/model_evo_harness/models' / framework / 'ctr_conditioning.py'
    assert path.exists(), f'Missing native conditioning module: {path}'
    return getattr(importlib.import_module(f'model_evo_harness.models.{framework}.ctr_conditioning'), name)


def tensor(framework, value):
    value = np.asarray(value, np.float32)
    if framework == 'pytorch':
        import torch
        return torch.tensor(value)
    import tensorflow as tf
    return tf.constant(value)


def array(value):
    return value.detach().numpy() if hasattr(value, 'detach') else value.numpy()


def assign(parameter, value):
    value = np.asarray(value, np.float32)
    if hasattr(parameter, 'assign'):
        parameter.assign(value)
    else:
        import torch
        with torch.no_grad(): parameter.copy_(torch.tensor(value))


def linear(layer, weight, bias):
    if hasattr(layer, 'kernel'): assign(layer.kernel, weight)
    else: assign(layer.weight, np.asarray(weight).T)
    assign(layer.bias, bias)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('rank,overparam', [(None, None), (2, None), (2, 3)])
def test_adaptive_linear_matches_condition_generated_matrix(framework, rank, overparam):
    model = cls(framework, 'AdaptiveLinear')(2, 2, 1, rank=rank, generated_bias=True, overparam_dim=overparam)
    x, c = tensor(framework, [[2, 3], [2, 3]]), tensor(framework, [[1], [2]])
    model(x, c)
    # Generated W(c)=diag(c,2*c), generated bias=(c,-c).
    linear(model.generator, [[1, 0, 0, 2, 1, -1]], np.zeros(6))
    if rank is not None:
        if overparam is None:
            assign(model.input_factor, np.eye(2)); assign(model.output_factor, np.eye(2))
        else:
            assign(model.input_left, [[1, 0, 0], [0, 1, 0]])
            assign(model.input_right, [[1, 0], [0, 1], [0, 0]])
            assign(model.output_left, [[1, 0, 0], [0, 1, 0]])
            assign(model.output_right, [[1, 0], [0, 1], [0, 0]])
    np.testing.assert_allclose(array(model(x, c)), [[3, 5], [6, 10]])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_prior_gate_matches_two_sigmoid_and_detaches_only_features(framework):
    model = cls(framework, 'PriorConditionedGate')(2, 1, 2, 2)
    h, x, p = tensor(framework, [[2, 4]]), tensor(framework, [[1, 2]]), tensor(framework, [[1]])
    model(h, x, p)
    linear(model.gate_in, [[1, 0], [0, 1], [1, 1]], [0, 0])
    linear(model.gate_out, np.eye(2), [0, 0])
    np.testing.assert_allclose(array(model(h, x, p)), [[4/(1+np.exp(-2)), 8/(1+np.exp(-3))]], atol=1e-6)
    for detach in (True, False):
        model.detach_features = detach
        if framework == 'pytorch':
            import torch
            inputs = [v.clone().requires_grad_() for v in (h, x, p)]
            model.zero_grad()
            model(*inputs).sum().backward()
            assert inputs[1].grad is None if detach else torch.count_nonzero(inputs[1].grad) > 0
            assert all(v.grad is not None and torch.count_nonzero(v.grad) for v in (inputs[0], inputs[2]))
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
        else:
            import tensorflow as tf
            inputs = [tf.Variable(v) for v in (h, x, p)]
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(model(*inputs))
            grads = tape.gradient(loss, inputs + model.trainable_variables)
            assert grads[1] is None if detach else np.count_nonzero(grads[1].numpy()) > 0
            assert all(v is not None and np.count_nonzero(v.numpy()) for v in (grads[0], grads[2]))
            assert all(v is not None and np.isfinite(v.numpy()).all() for v in grads[3:])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('rank,overparam', [(None, None), (2, None), (2, 4)])
def test_adaptive_shared_and_generated_parameters_receive_gradients(framework, rank, overparam):
    model = cls(framework, 'AdaptiveLinear')(3, 4, 2, rank=rank, overparam_dim=overparam)
    x, c = np.ones((2, 3), np.float32), np.array([[1, 2], [3, 4]], np.float32)
    if framework == 'pytorch':
        import torch
        x, c = torch.tensor(x, requires_grad=True), torch.tensor(c, requires_grad=True)
        model(x, c).square().sum().backward()
        grads = [v.grad for v in model.parameters()] + [x.grad, c.grad]
        assert all(v is not None and torch.isfinite(v).all() and torch.count_nonzero(v) for v in grads)
    else:
        import tensorflow as tf
        x, c = tf.Variable(x), tf.Variable(c)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(model(x, c) ** 2)
        grads = tape.gradient(loss, model.trainable_variables + [x, c])
        assert all(v is not None and np.isfinite(v.numpy()).all() and np.count_nonzero(v.numpy()) for v in grads)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('kwargs', [{'rank': 0}, {'rank': 4}, {'overparam_dim': 4}, {'rank': 2, 'overparam_dim': 1}])
def test_invalid_factorizations_rejected(framework, kwargs):
    with pytest.raises(ValueError): cls(framework, 'AdaptiveLinear')(2, 3, 1, **kwargs)


@pytest.mark.parametrize('rank,generated_bias,overparam', [(None, False, None), (None, True, None), (2, False, None), (2, True, 4)])
def test_adaptive_linear_mapped_framework_parameters_and_gradients(rank, generated_bias, overparam):
    import torch
    import tensorflow as tf
    torch.manual_seed(19)
    source = cls('pytorch', 'AdaptiveLinear')(3, 4, 2, rank, generated_bias, overparam)
    target = cls('tensorflow', 'AdaptiveLinear')(3, 4, 2, rank, generated_bias, overparam)
    values = [np.array([[1, 2, -1], [3, 1, .5]], np.float32), np.array([[1, 2], [3, -1]], np.float32)]
    tx, kx = [torch.tensor(x, requires_grad=True) for x in values], [tf.Variable(x) for x in values]
    target(*kx)
    linear(target.generator, source.generator.weight.detach().numpy().T, source.generator.bias.detach().numpy())
    for name, parameter in source.named_parameters(recurse=False):
        assign(getattr(target, name), parameter.detach().numpy())
    with tf.GradientTape() as tape:
        ky = target(*kx)
        loss = tf.reduce_sum(ky ** 2)
    ty = source(*tx)
    ty.square().sum().backward()
    np.testing.assert_allclose(array(ty), array(ky), rtol=2e-5, atol=2e-5)
    for actual, expected in zip(tape.gradient(loss, kx), tx):
        np.testing.assert_allclose(array(actual), array(expected.grad), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_conditioning_rejects_batch_broadcast_and_wrong_width(framework):
    adaptive = cls(framework, 'AdaptiveLinear')(2, 3, 2)
    prior = cls(framework, 'PriorConditionedGate')(2, 2, 3, 4)
    exceptions = (ValueError,)
    if framework == 'tensorflow':
        import tensorflow as tf
        exceptions += (tf.errors.InvalidArgumentError,)
    with pytest.raises(exceptions):
        adaptive(tensor(framework, np.ones((2, 2))), tensor(framework, np.ones((1, 2))))
    with pytest.raises(exceptions):
        prior(tensor(framework, np.ones((2, 3))), tensor(framework, np.ones((2, 2))), tensor(framework, np.ones((1, 2))))
    with pytest.raises(exceptions):
        adaptive(tensor(framework, np.ones((2, 3))), tensor(framework, np.ones((2, 2))))
