"""Analytic contracts for independently implemented CTR composition blocks."""
import importlib
from pathlib import Path

import numpy as np
import pytest


def cls(framework, name):
    path = Path(__file__).resolve().parents[1] / 'src/model_evo_harness/models' / framework / 'ctr_composition.py'
    assert path.exists(), f'Missing native composition module: {path}'
    return getattr(importlib.import_module(f'model_evo_harness.models.{framework}.ctr_composition'), name)


def tensor(framework, value):
    value = np.asarray(value, dtype=np.float32)
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
        with torch.no_grad():
            parameter.copy_(torch.tensor(value))


def linear(layer, weight, bias=None):
    if hasattr(layer, 'kernel'):
        assign(layer.kernel, weight)
    else:
        assign(layer.weight, np.asarray(weight).T)
    if bias is not None:
        assign(layer.bias, bias)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_gated_cross_keeps_initial_input_at_every_depth(framework):
    model = cls(framework, 'GatedCross')(2, 2)
    x = tensor(framework, [[2, 3]])
    model(x)
    for cross, gate in zip(model.cross_layers, model.gate_layers):
        linear(cross, np.eye(2), [1, -1])
        linear(gate, np.zeros((2, 2)))
    np.testing.assert_allclose(array(model(x)), [[11, 13.5]])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_mask_is_unbounded_and_original_input_can_drive_parallel_or_serial_blocks(framework):
    model = cls(framework, 'MaskBlock')(2, 2, 2, 2, layer_norm=False)
    original, hidden = tensor(framework, [[1, 2]]), tensor(framework, [[2, 3]])
    model(original, hidden)
    linear(model.mask_in, np.eye(2), [0, 0])
    linear(model.mask_out, np.eye(2), [2, -4])
    linear(model.projection, np.eye(2))
    np.testing.assert_allclose(array(model(original, hidden)), [[6, 0]])
    np.testing.assert_allclose(array(model(original, model(original, hidden))), [[18, 0]])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_context_feature_gate_has_two_sigmoid_scale(framework):
    model = cls(framework, 'ContextFeatureGate')(2, 2, 2)
    x, context = tensor(framework, [[2, -3]]), tensor(framework, [[np.log(3), 0]])
    model(x, context)
    linear(model.gate_in, np.eye(2), [0, 0])
    linear(model.gate_out, np.eye(2), [0, 0])
    np.testing.assert_allclose(array(model(x, context)), [[3, -3]], atol=1e-6)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_multihead_bilinear_excludes_cross_head_terms(framework):
    model = cls(framework, 'MultiHeadBilinearFusion')(4, 4, num_heads=2)
    x, y = tensor(framework, [[1, 2, 3, 4]]), tensor(framework, [[2, 3, 4, 5]])
    model(x, y)
    linear(model.left_linear, np.zeros((4, 1)), [0])
    linear(model.right_linear, np.zeros((4, 1)), [0])
    assign(model.interaction, np.stack([np.eye(2), np.eye(2)])[..., None])
    np.testing.assert_allclose(array(model(x, y)), [[40]])
    np.testing.assert_allclose(array(model(tensor(framework, [[1, 0, 0, 0]]),
                                          tensor(framework, [[0, 0, 1, 0]]))), [[0]])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_edcn_attention_uses_distinct_stream_parameters(framework):
    model = cls(framework, 'CrossDeepBridge')(2, mode='attention')
    x, y = tensor(framework, [[2, 3]]), tensor(framework, [[5, 7]])
    model(x, y)
    for gate in (model.cross_gate, model.deep_gate):
        layers = gate.layers if hasattr(gate, 'layers') else gate
        linear(layers[0], np.zeros((2, 2)), [0, 0])
        linear(layers[-1], np.zeros((2, 2)), [0, 0])
    deep_layers = model.deep_gate.layers if hasattr(model.deep_gate, 'layers') else model.deep_gate
    linear(deep_layers[-1], np.zeros((2, 2)), [np.log(3), 0])
    np.testing.assert_allclose(array(model(x, y)), [[4.75, 3.25]], atol=1e-6)
    for mode, expected in [('sum', [[7, 10]]), ('product', [[10, 21]])]:
        np.testing.assert_allclose(array(cls(framework, 'CrossDeepBridge')(2, mode)(x, y)), expected)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_field_regulation_shares_each_weight_over_embedding_coordinates(framework):
    model = cls(framework, 'FieldRegulation')(2, 2, temperature=2)
    x = tensor(framework, [[[2, 4], [6, 8]]])
    model(x)
    assign(model.cross_logits, [2 * np.log(3), 0])
    assign(model.deep_logits, [0, 2 * np.log(3)])
    cross, deep = model(x)
    np.testing.assert_allclose(array(cross), [[[1.5, 3], [1.5, 2]]], atol=1e-6)
    np.testing.assert_allclose(array(deep), [[[.5, 1], [4.5, 6]]], atol=1e-6)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_wukong_compressed_gram_and_field_projection(framework):
    model = cls(framework, 'WuKongBlock')(2, 2, linear_tokens=1, interaction_tokens=1,
                                         rank=1, hidden_dim=2, layer_norm=False)
    x = tensor(framework, [[[1, 2], [3, 4]]])
    model(x)
    assign(model.gram_projection, [[1], [0]])
    np.testing.assert_allclose(array(model.compressed_gram(x)), [[[5], [11]]])
    linear(model.fmb_in, np.zeros((2, 2)), [0, 0])
    linear(model.fmb_out, np.zeros((2, 2)), [2, 3])
    linear(model.linear_compression, [[1], [0]])
    np.testing.assert_allclose(array(model(x)), [[[3, 5], [4, 6]]])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('kind', ['vector', 'matrix'])
def test_field_pair_parameters_are_pair_specific(framework, kind):
    model = cls(framework, 'FieldMatrixInteraction')(3, 2, kind)
    x = tensor(framework, [[[1, 2], [3, 4], [5, 6]]])
    model(x)
    if kind == 'vector':
        assign(model.pair_weight, [[1, 2], [0, 1], [2, 0]])
    else:
        assign(model.pair_weight, [np.diag([1, 2]), np.diag([0, 1]), np.diag([2, 0])])
    np.testing.assert_allclose(array(model(x)), [[19, 12, 30]])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_grouped_interactions_count_within_and_across_once(framework):
    model = cls(framework, 'GroupedFieldInteraction')([0, 0, 1], 2)
    x = tensor(framework, [[[1, 2], [3, 4], [5, 6]]])
    model(x)
    assign(model.within_weights, [2, 3])
    assign(model.across_weights, [4])
    # 2*[3,8] + 4*([1,2]+[3,4])*[5,6]. Singleton group contributes zero.
    np.testing.assert_allclose(array(model(x)), [[86, 160]])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('mode,output_dim,expected', [('sum', 2, [[10, 18]]), ('concat', 4, [[2, 3, 8, 15]])])
def test_factorized_layer_affine_product(framework, mode, output_dim, expected):
    model = cls(framework, 'FactorizedInteraction')(2, output_dim, mode)
    x = tensor(framework, [[1, 1]])
    model(x)
    linear(model.projection, np.zeros((2, 4)), [2, 3, 4, 5])
    np.testing.assert_allclose(array(model(x)), expected)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_quadratic_heads_restore_original_field_coordinate_order(framework):
    model = cls(framework, 'QuadraticInteraction')(8, num_fields=2, num_heads=2, num_rows=2)
    x = tensor(framework, [np.arange(1, 9)])
    model(x)
    linear(model.projection, np.zeros((8, 16)), [1] * 8 + [2] * 8)
    # Two projected rows contribute 3*x; residual is x, with field/coordinate order preserved.
    np.testing.assert_allclose(array(model(x)), [4 * np.arange(1, 9)])
    linear(model.projection, np.zeros((8, 16)), [-1] * 16)
    np.testing.assert_allclose(array(model(x)), [np.arange(1, 9)])


CASES = [
    ('GatedCross', (4, 2), {}, [(2, 4)]),
    ('MaskBlock', (4, 3, 2, 5), {}, [(2, 4), (2, 3)]),
    ('ContextFeatureGate', (4, 3, 5), {}, [(2, 4), (2, 3)]),
    ('MultiHeadBilinearFusion', (4, 6, 2, 2), {}, [(2, 4), (2, 6)]),
    ('CrossDeepBridge', (4,), {'mode': 'attention'}, [(2, 4), (2, 4)]),
    ('CrossDeepBridge', (4,), {'mode': 'concat'}, [(2, 4), (2, 4)]),
    ('FieldRegulation', (3, 2), {}, [(2, 3, 2)]),
    ('WuKongBlock', (3, 4), {'linear_tokens': 2, 'interaction_tokens': 2}, [(2, 3, 4)]),
    ('FieldMatrixInteraction', (3, 2), {}, [(2, 3, 2)]),
    ('GroupedFieldInteraction', ([0, 0, 1], 3), {}, [(2, 3, 3)]),
    ('FactorizedInteraction', (4, 4), {}, [(2, 4)]),
    ('QuadraticInteraction', (8, 2, 2, 3), {}, [(2, 8)]),
]


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('name,args,kwargs,shapes', CASES)
def test_parameters_and_inputs_receive_finite_gradients(framework, name, args, kwargs, shapes):
    # Keep ReLU paths active reproducibly; arbitrary initializations may be all dead.
    if framework == 'pytorch':
        import torch
        torch.manual_seed(1)
    else:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(1)
    model = cls(framework, name)(*args, **kwargs)
    rng = np.random.default_rng(9)
    values = [rng.uniform(.2, 1.3, shape).astype('float32') for shape in shapes]
    if framework == 'pytorch':
        import torch
        inputs = [torch.tensor(x, requires_grad=True) for x in values]
        output = model(*inputs)
        outputs = output if isinstance(output, tuple) else (output,)
        loss = sum((v * torch.arange(1, v.numel()+1).reshape(v.shape)).sum() for v in outputs)
        loss.backward()
        gradients = [p.grad for p in model.parameters()] + [x.grad for x in inputs]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients[:-len(inputs)])
        assert all(torch.count_nonzero(g) for g in gradients[-len(inputs):])
    else:
        import tensorflow as tf
        inputs = [tf.Variable(x) for x in values]
        with tf.GradientTape() as tape:
            output = model(*inputs)
            outputs = output if isinstance(output, tuple) else (output,)
            loss = sum(tf.reduce_sum(v * tf.reshape(tf.cast(tf.range(1, tf.size(v)+1), v.dtype), tf.shape(v))) for v in outputs)
        gradients = tape.gradient(loss, model.trainable_variables + inputs)
        assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
        assert any(np.count_nonzero(g.numpy()) for g in gradients[:-len(inputs)])
        assert all(np.count_nonzero(g.numpy()) for g in gradients[-len(inputs):])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('name,args,kwargs', [
    ('GatedCross', (0,), {}), ('MaskBlock', (2, 0, 2, 2), {}),
    ('MultiHeadBilinearFusion', (3, 4), {'num_heads': 2}),
    ('CrossDeepBridge', (2,), {'mode': 'invalid'}),
    ('FieldRegulation', (2, 2), {'temperature': 0}),
    ('WuKongBlock', (2, 2), {'rank': 0}),
    ('FieldMatrixInteraction', (1, 2), {}),
    ('GroupedFieldInteraction', ([0, 2], 2), {}),
    ('FactorizedInteraction', (2, 3), {'residual_type': 'concat'}),
    ('QuadraticInteraction', (6, 2), {'num_heads': 2}),
])
def test_invalid_structural_dimensions_are_rejected(framework, name, args, kwargs):
    with pytest.raises(ValueError):
        cls(framework, name)(*args, **kwargs)


def _copy_native_weights(source, target):
    """Map equivalent parameter tensors, keeping assertions on real outputs."""
    import torch
    if isinstance(source, torch.nn.Linear):
        assign(target.kernel, source.weight.detach().numpy().T)
        if source.bias is not None:
            assign(target.bias, source.bias.detach().numpy())
    elif isinstance(source, torch.nn.LayerNorm):
        assign(target.gamma, source.weight.detach().numpy())
        assign(target.beta, source.bias.detach().numpy())
    elif isinstance(source, torch.nn.Sequential):
        dense = [v for v in source if isinstance(v, torch.nn.Linear)]
        for left, right in zip(dense, target.layers):
            _copy_native_weights(left, right)
    elif isinstance(source, torch.nn.ModuleList):
        for left, right in zip(source, target):
            _copy_native_weights(left, right)
    else:
        for name, parameter in source.named_parameters(recurse=False):
            assign(getattr(target, name), parameter.detach().numpy())
        for name, child in source.named_children():
            _copy_native_weights(child, getattr(target, name))


@pytest.mark.parametrize('name,args,kwargs,shapes', CASES)
def test_mapped_framework_weights_agree_on_values_and_input_gradients(name, args, kwargs, shapes):
    import torch
    import tensorflow as tf
    torch.manual_seed(17)
    source = cls('pytorch', name)(*args, **kwargs)
    target = cls('tensorflow', name)(*args, **kwargs)
    rng = np.random.default_rng(31)
    data = [rng.normal(size=shape).astype('float32') for shape in shapes]
    tx = [torch.tensor(x, requires_grad=True) for x in data]
    kx = [tf.Variable(x) for x in data]
    target(*kx)
    _copy_native_weights(source, target)
    with tf.GradientTape() as tape:
        ky = target(*kx)
        ky = ky if isinstance(ky, tuple) else (ky,)
        kloss = sum(tf.reduce_sum(v ** 2) for v in ky)
    ty = source(*tx)
    ty = ty if isinstance(ty, tuple) else (ty,)
    sum(v.square().sum() for v in ty).backward()
    kg = tape.gradient(kloss, kx)
    for left, right in zip(ty, ky):
        np.testing.assert_allclose(array(left), array(right), atol=3e-5, rtol=3e-5)
    for left, right in zip(tx, kg):
        np.testing.assert_allclose(array(left.grad), array(right), atol=8e-5, rtol=8e-5)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('name,args,kwargs,shapes', CASES)
def test_wrong_width_or_field_axis_is_rejected(framework, name, args, kwargs, shapes):
    model = cls(framework, name)(*args, **kwargs)
    values = [np.ones(shape, np.float32) for shape in shapes]
    values[0] = np.ones((*shapes[0][:-1], shapes[0][-1] + 1), np.float32)
    exceptions = (ValueError,)
    if framework == 'tensorflow':
        import tensorflow as tf
        exceptions += (tf.errors.InvalidArgumentError,)
    with pytest.raises(exceptions):
        model(*[tensor(framework, value) for value in values])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_fusion_rejects_batch_broadcast(framework):
    model = cls(framework, 'MultiHeadBilinearFusion')(2, 2)
    exceptions = (ValueError,)
    if framework == 'tensorflow':
        import tensorflow as tf
        exceptions += (tf.errors.InvalidArgumentError,)
    with pytest.raises(exceptions):
        model(tensor(framework, np.ones((2, 2))), tensor(framework, np.ones((1, 2))))
