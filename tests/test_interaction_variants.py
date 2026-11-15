"""Analytic native interaction checks, optionally required on a real GPU."""

import importlib
import os

import numpy as np
import pytest


@pytest.fixture(params=['pytorch', 'tensorflow'])
def backend(request):
    name = request.param
    required = os.environ.get('MODELEVO_REQUIRE_GPU') == '1'
    try:
        lib = importlib.import_module('torch' if name == 'pytorch' else 'tensorflow')
    except ImportError:
        if required:
            pytest.fail(f'{name} required')
        pytest.skip(f'{name} unavailable')
    available = lib.cuda.is_available() if name == 'pytorch' else bool(lib.config.list_physical_devices('GPU'))
    if required and not available:
        pytest.fail(f'{name} GPU required')
    if name == 'tensorflow':
        with lib.device('/GPU:0' if required else '/CPU:0'):
            yield name
    else:
        yield name


def cls(framework, module, name):
    return getattr(importlib.import_module(f'model_evo_harness.models.{framework}.{module}'), name)


def tensor(framework, x, dtype='float32'):
    if framework == 'pytorch':
        import torch
        return torch.tensor(np.asarray(x, dtype=dtype), device='cuda:0' if os.environ.get('MODELEVO_REQUIRE_GPU') == '1' else 'cpu')
    import tensorflow as tf
    return tf.constant(np.asarray(x, dtype=dtype))


def layer(framework, module, name, *args, **kwargs):
    obj = cls(framework, module, name)(*args, **kwargs)
    return obj.to('cuda:0') if framework == 'pytorch' and os.environ.get('MODELEVO_REQUIRE_GPU') == '1' else obj


def array(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else x.numpy()


def assign(framework, target, value):
    if framework == 'pytorch':
        import torch
        with torch.no_grad():
            target.copy_(tensor(framework, value))
    else:
        target.assign(value)


def gradients(framework, obj, inputs):
    if framework == 'pytorch':
        output = obj(*inputs)
        output.square().sum().backward()
        params = list(obj.parameters())
        grads = [p.grad for p in params]
    else:
        import tensorflow as tf
        with tf.GradientTape() as tape:
            output = obj(*inputs)
            loss = tf.reduce_sum(output ** 2)
        params = list(obj.trainable_variables)
        grads = tape.gradient(loss, params)
    assert params and all(g is not None for g in grads)
    for value in [output, *params, *grads]:
        assert np.isfinite(array(value)).all()
        if os.environ.get('MODELEVO_REQUIRE_GPU') == '1':
            if hasattr(value, 'is_cuda'):
                assert value.is_cuda
            else:
                assert 'GPU:0' in (value.device if hasattr(value, 'device') else value.handle.device)
    assert any(np.any(array(g) != 0) for g in grads)
    return output


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def test_cross_mix_matches_multilayer_nonlinear_expert_equation(backend):
    obj = layer(backend, 'interaction_variants', 'CrossNetMix', 3, rank=2, num_experts=2, num_layers=2)
    values = {}
    rng = np.random.default_rng(11)
    for name, shape in [('u', (2, 2, 3, 2)), ('v', (2, 2, 3, 2)), ('c', (2, 2, 2, 2)), ('bias', (2, 3)), ('router', (3, 2))]:
        values[name] = rng.normal(size=shape).astype('float32') * .4
        assign(backend, getattr(obj, name), values[name])
    x = np.array([[.3, -.4, .9], [-.2, .7, .1]], dtype='float32')
    h, expected_routing = x.copy(), []
    for index in range(2):
        weights = softmax(h @ values['router'])
        expected_routing.append(weights)
        low = np.tanh(np.einsum('bd,edr->ber', h, values['v'][index]))
        low = np.tanh(np.einsum('ers,bes->ber', values['c'][index], low))
        expert = np.einsum('edr,ber->bed', values['u'][index], low)
        h = h + x * (np.einsum('be,bed->bd', weights, expert) + values['bias'][index])
    result, routing = obj(tensor(backend, x), return_routing=True)
    np.testing.assert_allclose(array(result), h, atol=2e-6)
    np.testing.assert_allclose(array(routing), np.stack(expected_routing, axis=1), atol=2e-6)
    np.testing.assert_allclose(array(routing).sum(-1), 1., atol=1e-6)
    gradients(backend, obj, [tensor(backend, x)])


def test_cross_mix_zero_projection_is_identity(backend):
    obj = layer(backend, 'interaction_variants', 'CrossNetMix', 2, rank=1, num_experts=1, num_layers=2)
    assign(backend, obj.u, np.zeros((2, 1, 2, 1)))
    x = tensor(backend, [[2., -3.]])
    result, weights = obj(x, return_routing=True)
    np.testing.assert_array_equal(array(result), array(x))
    np.testing.assert_array_equal(array(weights), np.ones((1, 2, 1)))


@pytest.mark.parametrize('mode', ['ifm', 'difm'])
def test_input_aware_factor_jointly_weights_linear_and_pair_terms(backend, mode):
    obj = layer(backend, 'interaction_variants', 'InputAwareFM', 3, 2, hidden_dims=(4,), mode=mode, num_heads=1)
    x = tensor(backend, [[[1., .2], [.3, -.4], [.7, .8]], [[.4, -.5], [-.2, .8], [.5, .6]]])
    linear = tensor(backend, [[.2, .6, -.3], [-.1, .5, .9]])
    factors = array(obj.field_factors(x))
    if mode == 'ifm':
        np.testing.assert_allclose(factors.sum(-1), 3., atol=1e-6)
        assert np.all(factors > 0)
    else:
        bit, vector = obj.factor_components(x)
        np.testing.assert_allclose(factors, array(bit) + array(vector), atol=1e-6)
    weighted = array(x) * factors[..., None]
    expected = (array(linear) * factors).sum(-1)
    expected += sum((weighted[:, i] * weighted[:, j]).sum(-1) for i, j in [(0, 1), (0, 2), (1, 2)])
    np.testing.assert_allclose(array(obj(x, linear)), expected, atol=2e-5)
    gradients(backend, obj, [x, linear])


def test_difm_factors_are_not_forced_positive_or_normalized(backend):
    obj = layer(backend, 'interaction_variants', 'InputAwareFM', 2, 2, hidden_dims=(), mode='difm', num_heads=1)
    x = tensor(backend, [[[1., 0.], [0., 1.]]])
    obj.field_factors(x)
    assign(backend, obj.bit_output, -np.ones((4, 2), dtype='float32'))
    assign(backend, obj.vector_output, np.zeros((4, 2), dtype='float32'))
    np.testing.assert_allclose(array(obj.field_factors(x)), [[-2., -2.]], atol=1e-6)


def test_onn_vector_mode_keeps_coordinates_and_ignores_copy(backend):
    values = np.arange(1, 19, dtype='float32').reshape(1, 3, 3, 2)
    obj = layer(backend, 'field_products', 'OperationAwareInteraction', 3, 2, output_mode='pair_vectors')
    expected = np.stack([values[:, i, j] * values[:, j, i] for i, j in [(0, 1), (0, 2), (1, 2)]], axis=1)
    np.testing.assert_array_equal(array(obj(tensor(backend, values))), expected)
    values[:, np.arange(3), np.arange(3)] = -100
    np.testing.assert_array_equal(array(obj(tensor(backend, values))), expected)
    default = layer(backend, 'field_products', 'OperationAwareInteraction', 3, 2)
    np.testing.assert_array_equal(array(default(tensor(backend, values)))[:, -3:], expected.sum(-1))
    mask = tensor(backend, [[True, False, True]], 'bool')
    values[:, 1] = np.nan
    values[:, :, 1] = np.nan
    masked = array(obj(tensor(backend, values), valid_mask=mask))
    np.testing.assert_array_equal(masked[:, [0, 2]], 0.)
    np.testing.assert_array_equal(masked[:, 1], expected[:, 1])


def test_symmetric_field_matrix_matches_fefm_and_reversal(backend):
    obj = layer(backend, 'ctr_composition', 'FieldMatrixInteraction', 2, 2, symmetric=True)
    matrix = np.array([[[1., 2.], [3., 4.]]], dtype='float32')
    assign(backend, obj.pair_weight, matrix)
    values = np.array([[[1., .5], [-.2, .7]]], dtype='float32')
    expected = np.einsum('bi,ij,bj->b', values[:, 0], matrix[0] + matrix[0].T, values[:, 1])
    np.testing.assert_allclose(array(obj(tensor(backend, values)))[:, 0], expected, atol=1e-6)
    np.testing.assert_allclose(array(obj(tensor(backend, values[:, ::-1].copy())))[:, 0], expected, atol=1e-6)
    gradients(backend, obj, [tensor(backend, values)])
    with pytest.raises(ValueError):
        layer(backend, 'ctr_composition', 'FieldMatrixInteraction', 2, 2, interaction_type='vector', symmetric=True)


@pytest.mark.parametrize('kind,count', [('all', 1), ('field', 2), ('pair', 3)])
def test_fibinet_sharing_and_branch_weights(backend, kind, count):
    obj = layer(backend, 'interactions', 'FiBiNET', [2, 2, 2], 2, bilinear_type=kind, separate_branches=True)
    raw = tensor(backend, [[[1., 2.], [3., 4.], [5., 6.]]])
    obj.bilinear_features(raw)
    obj.bilinear_features(raw, gated=True)
    weights = np.stack([(i+1) * np.eye(2) for i in range(count)]).astype('float32')
    raw_layers = [obj.bilinear] if kind == 'all' else list(obj.bilinear)
    se_layers = [obj.gated_bilinear] if kind == 'all' else list(obj.gated_bilinear)
    for i, (left, right) in enumerate(zip(raw_layers, se_layers)):
        assign(backend, left.weight if backend == 'pytorch' else left.kernel, weights[i])
        assign(backend, right.weight if backend == 'pytorch' else right.kernel, weights[i] * 2)
    expected = np.stack([array(raw)[:, i] * array(raw)[:, j] * (1 if kind == 'all' else (i+1 if kind == 'field' else index+1)) for index, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)])], 1)
    np.testing.assert_allclose(array(obj.bilinear_features(raw)), expected, atol=1e-6)
    np.testing.assert_allclose(array(obj.bilinear_features(raw, gated=True)), 2 * expected, atol=1e-6)
    result = obj(tensor(backend, [[0, 1, 0]], 'int64'))
    assert array(result).shape == (1,)


@pytest.mark.parametrize('mode', ['ifm', 'difm'])
def test_field_factors_match_fixed_multihead_numpy_equation(backend, mode):
    obj = layer(backend, 'interaction_variants', 'InputAwareFM', 3, 4, hidden_dims=(), mode=mode, num_heads=2)
    rng = np.random.default_rng(47)
    tokens = rng.normal(size=(2, 3, 4)).astype('float32') * .2
    bit_output = rng.normal(size=(12, 3)).astype('float32')
    assign(backend, obj.bit_output, bit_output)
    bit = tokens.reshape(2, 12) @ bit_output
    if mode == 'ifm':
        expected = 3 * softmax(bit)
    else:
        projections = []
        for name in ['query', 'key', 'value']:
            value = rng.normal(size=(4, 4)).astype('float32') * .3
            assign(backend, getattr(obj, name), value)
            projections.append((tokens @ value).reshape(2, 3, 2, 2).transpose(0, 2, 1, 3))
        q, k, v = projections
        residual = rng.normal(size=(4, 4)).astype('float32') * .3
        vector_output = rng.normal(size=(12, 3)).astype('float32')
        assign(backend, obj.residual, residual)
        assign(backend, obj.vector_output, vector_output)
        attended = (softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(2)) @ v).transpose(0, 2, 1, 3).reshape(2, 3, 4)
        vector = np.maximum(attended + tokens @ residual, 0).reshape(2, 12) @ vector_output
        expected = bit + vector
    np.testing.assert_allclose(array(obj.field_factors(tensor(backend, tokens))), expected, atol=2e-6)


def test_cross_mix_input_gradient_matches_finite_difference(backend):
    obj = layer(backend, 'interaction_variants', 'CrossNetMix', 2, rank=1, num_experts=2, num_layers=2)
    for name, shape in [('u', (2, 2, 2, 1)), ('v', (2, 2, 2, 1)), ('c', (2, 2, 1, 1)), ('router', (2, 2))]:
        values = np.linspace(-.4, .6, np.prod(shape), dtype='float32').reshape(shape)
        assign(backend, getattr(obj, name), values)
    x = tensor(backend, [[.3, -.2]])
    if backend == 'pytorch':
        x.requires_grad_(True)
        obj(x).sum().backward()
        gradient = array(x.grad)
    else:
        import tensorflow as tf
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.reduce_sum(obj(x))
        gradient = array(tape.gradient(loss, x))
    numerical = np.zeros((1, 2))
    for index in range(2):
        upper, lower = array(x).copy(), array(x).copy()
        upper[0, index] += .001
        lower[0, index] -= .001
        numerical[0, index] = (array(obj(tensor(backend, upper))).sum() - array(obj(tensor(backend, lower))).sum()) / .002
    np.testing.assert_allclose(gradient, numerical, atol=3e-5, rtol=3e-4)


def test_dual_entry_point_cannot_silently_select_ifm(backend):
    obj = layer(backend, 'interaction_variants', 'DualInputAwareFM', 3, 4, num_heads=2)
    assert obj.mode == 'difm'
    assert array(obj(tensor(backend, np.ones((2, 3, 4))), tensor(backend, np.ones((2, 3))))).shape == (2,)


@pytest.mark.parametrize('name,args,kwargs', [
    ('CrossNetMix', (0,), {}),
    ('CrossNetMix', (2,), {'num_experts': 0}),
    ('InputAwareFM', (1, 2), {}),
    ('InputAwareFM', (3, 3), {'mode': 'difm', 'num_heads': 2}),
    ('InputAwareFM', (3, 2), {'mode': 'invalid'}),
])
def test_invalid_variant_configurations_fail(backend, name, args, kwargs):
    with pytest.raises(ValueError):
        layer(backend, 'interaction_variants', name, *args, **kwargs)
