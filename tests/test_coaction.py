"""Numeric parameter layout, masking and gradient contracts for native CoAction."""
import importlib
from pathlib import Path

import numpy as np
import pytest


def coaction(framework):
    path = Path(__file__).resolve().parents[1] / 'src/model_evo_harness/models' / framework / 'coaction.py'
    assert path.is_file(), f'Missing native CoAction implementation: {path}'
    return importlib.import_module(f'model_evo_harness.models.{framework}.coaction').CoAction


def tensor(framework, value):
    value = np.asarray(value)
    if framework == 'pytorch':
        import torch
        return torch.tensor(value)
    import tensorflow as tf
    return tf.constant(value)


def array(value):
    return value.detach().numpy() if hasattr(value, 'detach') else value.numpy()


def fixture():
    # Per sample: row-major W1[2,2], b1[2], W2[2,1], b2[1].
    parameters = np.array([[.5, -1, 1, .25, .2, -.3, 2, -.5, .4],
                           [-.25, .5, .75, -1, -.1, .2, -1, 1.5, -.2]], np.float32)
    feed = np.array([[[.5, -1], [1.5, .25], [np.nan, np.nan]],
                     [[.5, -1], [np.nan, np.nan], [-.5, .75]]], np.float32)
    valid = np.array([[True, True, False], [True, False, True]])
    return parameters, feed, valid


def oracle(feed, valid, reduction):
    # Scalar equations specify both samples independently of parameter slicing.
    result = np.zeros((2, 3, 9), np.float32)
    for batch, time in zip(*np.where(valid)):
        x, y = feed[batch, time]
        for power in (1, 2, 3):
            if batch == 0:
                a = np.tanh(.5*x**power + y**power + .2)
                b = np.tanh(-x**power + .25*y**power - .3)
                z = 2*a - .5*b + .4
            else:
                a = np.tanh(-.25*x**power + .75*y**power - .1)
                b = np.tanh(.5*x**power - y**power + .2)
                z = -a + 1.5*b - .2
            result[batch, time, 3*(power-1):3*power] = [a, b, z]
    return result if reduction == 'concat' else result.reshape(2, 3, 3, 3).sum(2)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('reduction', ['concat', 'sum'])
def test_multilayer_orders_share_each_samples_parameters_and_read_every_layer(framework, reduction):
    model = coaction(framework)(2, [2, 1], order=3, order_reduction=reduction, use_bias=True)
    values = fixture()
    result = model(*(tensor(framework, value) for value in values))
    assert model.parameter_dim == 9
    assert model.output_dim == (9 if reduction == 'concat' else 3)
    np.testing.assert_allclose(array(result), oracle(values[1], values[2], reduction), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_default_has_no_bias_and_final_layer_remains_linear(framework):
    model = coaction(framework)(2, [1])
    values = (np.array([[2, -3], [-1, 4]], np.float32),
              np.array([[[1, 2], [-2, 1]], [[1, 2], [-2, 1]]], np.float32),
              np.ones((2, 2), bool))
    assert model.parameter_dim == 2
    assert model.output_dim == 1
    np.testing.assert_allclose(array(model(*(tensor(framework, v) for v in values))), [[[-4], [-7]], [[7], [6]]])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('length', [0, 3])
def test_empty_and_nan_padded_rows_have_zero_outputs_and_finite_zero_gradients(framework, length):
    model = coaction(framework)(2, [2, 1], order=3, use_bias=True)
    parameters, feed, valid = fixture()
    parameters[1] = np.nan
    feed[1] = np.nan
    valid[1] = False
    feed, valid = feed[:, :length], valid[:, :length]
    if framework == 'pytorch':
        import torch
        parameters = torch.tensor(parameters, requires_grad=True)
        feed = torch.tensor(feed, requires_grad=True)
        result = model(parameters, feed, torch.tensor(valid))
        result.square().sum().backward()
        gradients = [parameters.grad, feed.grad]
    else:
        import tensorflow as tf
        parameters, feed = tf.Variable(parameters), tf.Variable(feed)
        with tf.GradientTape() as tape:
            result = model(parameters, feed, tf.constant(valid))
            loss = tf.reduce_sum(result ** 2)
        gradients = tape.gradient(loss, [parameters, feed])
    assert array(result).shape == (2, length, 9)
    assert np.isfinite(array(result)).all()
    np.testing.assert_array_equal(array(result)[~valid], 0)
    assert all(g is not None and np.isfinite(array(g)).all() for g in gradients)
    np.testing.assert_array_equal(array(gradients[0])[1], 0)
    np.testing.assert_array_equal(array(gradients[1])[~valid], 0)
    if length:
        assert np.count_nonzero(array(gradients[0])[0])
        assert np.count_nonzero(array(gradients[1])[valid])
    else:
        np.testing.assert_array_equal(array(gradients[0]), 0)


@pytest.mark.parametrize('reduction', ['concat', 'sum'])
@pytest.mark.parametrize('use_bias', [False, True])
def test_framework_outputs_and_gradients_match_with_trainable_host_inputs(reduction, use_bias):
    import torch
    import tensorflow as tf
    source = coaction('pytorch')(2, [3, 2, 1], order=3, order_reduction=reduction, use_bias=use_bias)
    target = coaction('tensorflow')(2, [3, 2, 1], order=3, order_reduction=reduction, use_bias=use_bias)
    rng = np.random.default_rng(72)
    parameters = rng.normal(0, .3, (2, source.parameter_dim)).astype(np.float32)
    _, feed, valid = fixture()
    tx = [torch.tensor(v, requires_grad=True) for v in (parameters, feed)]
    kx = [tf.Variable(v) for v in (parameters, feed)]
    weights = np.arange(1, 1 + 2*3*source.output_dim, dtype=np.float32).reshape(2, 3, -1) / 10
    ty = source(*tx, torch.tensor(valid))
    (ty * torch.tensor(weights)).sum().backward()
    with tf.GradientTape() as tape:
        ky = target(*kx, tf.constant(valid))
        loss = tf.reduce_sum(ky * weights)
    gradients = tape.gradient(loss, kx)
    np.testing.assert_allclose(array(ty), array(ky), rtol=2e-5, atol=2e-5)
    for expected, actual in zip(tx, gradients):
        assert actual is not None and np.isfinite(array(actual)).all()
        assert expected.grad is not None and np.isfinite(array(expected.grad)).all()
        assert np.count_nonzero(array(actual))
        np.testing.assert_allclose(array(expected.grad), array(actual), rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(array(tx[1].grad)[~valid], 0)
    # The host owns the learnable tables; the component introduces no shared weights.
    assert not list(source.parameters())
    assert not target.trainable_variables


def test_tensorflow_dynamic_graph_handles_empty_and_variable_lengths():
    import tensorflow as tf
    model = coaction('tensorflow')(2, [2, 1], order=3, use_bias=True)
    @tf.function(autograph=False, input_signature=[tf.TensorSpec([None, 9], tf.float32),
                                                  tf.TensorSpec([None, None, 2], tf.float32),
                                                  tf.TensorSpec([None, None], tf.bool)])
    def run(parameters, feed, valid):
        return model(parameters, feed, valid)
    parameters, feed, valid = fixture()
    for length in (3, 0, 1):
        result = run(parameters, feed[:, :length], valid[:, :length])
        assert result.shape == (2, length, 9)
        np.testing.assert_allclose(array(result), oracle(feed, valid, 'concat')[:, :length], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('kwargs', [dict(input_dim=0), dict(input_dim=True), dict(input_dim=1.5),
                                   dict(layer_dims=[]), dict(layer_dims=[2, 0]), dict(layer_dims=[True]),
                                   dict(order=0), dict(order=True), dict(order=1.5),
                                   dict(order_reduction='mean'), dict(use_bias=1)])
def test_invalid_configuration_rejected(framework, kwargs):
    config = dict(input_dim=2, layer_dims=[2, 1])
    config.update(kwargs)
    with pytest.raises(ValueError):
        coaction(framework)(**config)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('problem', ['parameter_rank', 'parameter_width', 'feed_rank', 'feed_width',
                                    'mask_rank', 'mask_shape', 'batch', 'mask_dtype',
                                    'parameter_dtype', 'feed_dtype', 'mixed_dtype'])
def test_invalid_inputs_rejected_without_broadcast_or_implicit_dtype_conversion(framework, problem):
    model = coaction(framework)(2, [2, 1], order=3, use_bias=True)
    parameters, feed, valid = fixture()
    if problem == 'parameter_rank': parameters = parameters[:, None]
    if problem == 'parameter_width': parameters = parameters[:, :-1]
    if problem == 'feed_rank': feed = feed[:, 0]
    if problem == 'feed_width': feed = feed[..., :1]
    if problem == 'mask_rank': valid = valid[..., None]
    if problem == 'mask_shape': valid = valid[:, :1]
    if problem == 'batch': parameters = parameters[:1]
    if problem == 'mask_dtype': valid = valid.astype(np.int32)
    if problem == 'parameter_dtype': parameters = parameters.astype(np.int32)
    if problem == 'feed_dtype': feed = np.zeros(feed.shape, np.int32)
    if problem == 'mixed_dtype': parameters = parameters.astype(np.float64)
    exceptions = (ValueError, TypeError)
    if framework == 'tensorflow':
        import tensorflow as tf
        exceptions += (tf.errors.InvalidArgumentError,)
    with pytest.raises(exceptions):
        model(*(tensor(framework, v) for v in (parameters, feed, valid)))
