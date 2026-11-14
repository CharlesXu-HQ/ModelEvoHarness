"""PEP equations, threshold sharing, and fixed-mask lottery-ticket retraining."""
import importlib
from pathlib import Path

import numpy as np
import pytest


def layer_class(framework):
    path = Path(__file__).resolve().parents[1] / 'src/model_evo_harness/models' / framework / 'embedding_pruning.py'
    assert path.exists(), f'Missing native PEP mechanism: {path}'
    return getattr(importlib.import_module(f'model_evo_harness.models.{framework}.embedding_pruning'),
                   'PrunableEmbedding')


def array(value):
    return value.detach().numpy() if hasattr(value, 'detach') else value.numpy()


def ids(framework, value):
    if framework == 'pytorch':
        import torch
        return torch.tensor(value, dtype=torch.int64)
    import tensorflow as tf
    return tf.constant(value, dtype=tf.int64)


def assign(parameter, value):
    if hasattr(parameter, 'assign'):
        parameter.assign(value)
    else:
        import torch
        with torch.no_grad():
            parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype))


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_soft_threshold_shrinks_survivors_and_produces_exact_zeros(framework):
    weights = np.array([[-.7, -.5, -.1, 0., .5, .8], [9.] * 6], np.float64)
    layer = layer_class(framework)(weights, threshold_init=0., padding_idx=1)
    actual = array(layer(ids(framework, [[0, 1], [1, 0]])))
    expected = [[-.2, 0., 0., 0., 0., .3], [0.] * 6]
    np.testing.assert_allclose(actual, np.array(expected)[[[0, 1], [1, 0]]], atol=1e-15)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(array(layer.export_mask()),
                                  [[True, False, False, False, False, True], [False] * 6])
    assert int(array(layer.nonzero_count())) == 2
    assert tuple(layer(ids(framework, [])).shape) == (0, 6)
    assert tuple(layer(ids(framework, 0)).shape) == (6,)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('kind,logits,expected', [
    ('global', [[0.]], [[1.5] * 3] * 2),
    ('dimension', [[-np.log(3), 0., np.log(3)]], [[1.75, 1.5, 1.25]] * 2),
    ('feature', [[-np.log(3)], [np.log(3)]], [[1.75] * 3, [1.25] * 3]),
    ('feature_dim', [[-np.log(3), 0., np.log(3)], [np.log(3), -np.log(3), 0.]],
     [[1.75, 1.5, 1.25], [1.25, 1.75, 1.5]]),
])
def test_threshold_granularity_broadcasts_without_field_count_heuristics(framework, kind, logits, expected):
    layer = layer_class(framework)(np.full((2, 3), 2., np.float64), threshold_type=kind)
    assign(layer.threshold_logits, logits)
    np.testing.assert_allclose(array(layer(ids(framework, [0, 1]))), expected, atol=1e-15)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_search_gradients_match_analytic_values_with_repeated_ids_and_padding(framework):
    layer = layer_class(framework)(np.array([[.8, -.9, .2], [-.7, .5, -.4], [9., 9., 9.]],
                                            np.float64), threshold_init=0., padding_idx=2)
    selected = ids(framework, [[0, 1], [1, 2]])
    if framework == 'pytorch':
        layer(selected).square().sum().backward()
        weight_grad, threshold_grad = layer.weight.grad, layer.threshold_logits.grad
    else:
        import tensorflow as tf
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(selected) ** 2)
        weight_grad, threshold_grad = tape.gradient(loss, [layer.weight, layer.threshold_logits])
        weight_grad = tf.convert_to_tensor(weight_grad)
    np.testing.assert_allclose(array(weight_grad), [[.6, -.8, 0.], [-.8, 0., 0.], [0., 0., 0.]], atol=1e-15)
    np.testing.assert_allclose(array(threshold_grad), [[-.55]], atol=1e-15)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_retraining_rewinds_initial_weights_and_never_reactivates_pruned_entries(framework):
    cls = layer_class(framework)
    initial = np.array([[.6, -.8, .2], [.4, 0., .1], [9., 9., 9.]], np.float64)
    search = cls(initial, threshold_init=0., padding_idx=2)
    assign(search.weight, [[.7, -.1, .9], [-.8, .6, .2], [9., 9., 9.]])
    mask = search.export_mask()
    retrain = cls.from_mask(initial, mask, padding_idx=2)
    expected_mask = [[True, False, True], [True, True, False], [False, False, False]]
    np.testing.assert_array_equal(array(retrain.export_mask()), expected_mask)
    np.testing.assert_allclose(array(retrain.effective_weight()), [[.6, 0., .2], [.4, 0., 0.], [0., 0., 0.]])
    assert retrain.threshold_logits is None
    assert int(array(retrain.nonzero_count())) == 3  # Active zero stays trainable.
    selected = ids(framework, [0, 1, 2])
    if framework == 'pytorch':
        import torch
        optimizer = torch.optim.SGD(retrain.parameters(), lr=.1, momentum=.9, weight_decay=.1)
        retrain(selected).sum().backward()
        gradient = array(retrain.weight.grad)
        optimizer.step()
        assert len(list(retrain.parameters())) == 1
    else:
        import tensorflow as tf
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(retrain(selected))
        grad = tape.gradient(loss, retrain.weight)
        gradient = array(tf.convert_to_tensor(grad))
        tf.keras.optimizers.SGD(.1, momentum=.9, weight_decay=.1).apply_gradients([(grad, retrain.weight)])
        assert len(retrain.trainable_variables) == 1
    np.testing.assert_array_equal(gradient, expected_mask)
    assert array(retrain.effective_weight())[1, 1] < 0  # Zero active weight receives a gradient.
    np.testing.assert_array_equal(array(retrain.export_mask()), expected_mask)
    # Even an external raw-weight update cannot resurrect masked coordinates.
    assign(retrain.weight, np.ones((3, 3)))
    np.testing.assert_array_equal(array(retrain.effective_weight()), expected_mask)
    np.testing.assert_array_equal(initial, [[.6, -.8, .2], [.4, 0., .1], [9., 9., 9.]])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_all_pruned_retraining_is_finite_zero_with_zero_gradients(framework):
    cls = layer_class(framework)
    layer = cls.from_mask([[1., 2.]], [[False, False]])
    if framework == 'pytorch':
        layer(ids(framework, [0, 0])).sum().backward()
        grad = layer.weight.grad
    else:
        import tensorflow as tf
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(ids(framework, [0, 0])))
        grad = tf.convert_to_tensor(tape.gradient(loss, layer.weight))
    np.testing.assert_array_equal(array(layer.effective_weight()), [[0., 0.]])
    np.testing.assert_array_equal(array(grad), [[0., 0.]])
    assert int(array(layer.nonzero_count())) == 0


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
def test_fixed_mask_is_preserved_by_framework_weight_checkpoints(framework):
    cls = layer_class(framework)
    source = cls.from_mask([[1., 2.], [3., 4.]], [[True, False], [False, True]], padding_idx=1)
    target = cls.from_mask([[0., 0.], [0., 0.]], [[True, True], [True, True]])
    if framework == 'pytorch':
        target.load_state_dict(source.state_dict())
    else:
        target.set_weights(source.get_weights())
    np.testing.assert_array_equal(array(target(ids(framework, [0, 1]))), [[1., 0.], [0., 0.]])
    np.testing.assert_array_equal(array(target.export_mask()), [[True, False], [False, False]])


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('weights,kwargs', [
    ([], {}), ([[1., np.nan]], {}), ([[1j]], {}), ([[1.]], {'threshold_type': 'field'}),
    ([[1.]], {'threshold_init': np.inf}), ([[1.]], {'threshold_init': True}),
    ([[1.]], {'padding_idx': True}), ([[1.]], {'padding_idx': 1}),
])
def test_constructor_rejects_invalid_table_or_configuration(framework, weights, kwargs):
    with pytest.raises((ValueError, TypeError)):
        layer_class(framework)(weights, **kwargs)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('mask', [[[1, 0]], [[True]], [[True, False], [False, True]]])
def test_retraining_requires_exact_boolean_mask_shape(framework, mask):
    with pytest.raises((ValueError, TypeError)):
        layer_class(framework).from_mask([[1., 2.]], mask)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('value', [[-1], [2], [.5]])
def test_invalid_ids_do_not_alias_vocabulary_rows(framework, value):
    layer = layer_class(framework)([[1.], [2.]])
    if framework == 'pytorch':
        import torch
        invalid = torch.tensor(value)
        error = ValueError
    else:
        import tensorflow as tf
        invalid = tf.constant(value)
        error = (ValueError, tf.errors.InvalidArgumentError)
    with pytest.raises(error):
        layer(invalid)


def test_tensorflow_lookup_works_in_graph_with_runtime_id_validation():
    import tensorflow as tf
    layer = layer_class('tensorflow')([[.8, -.9], [.1, .5]], threshold_init=0.)
    lookup = tf.function(layer, input_signature=[tf.TensorSpec([None], tf.int64)])
    np.testing.assert_allclose(lookup(tf.constant([0, 1], tf.int64)).numpy(), [[.3, -.4], [0., 0.]], atol=1e-7)
    with pytest.raises(tf.errors.InvalidArgumentError):
        lookup(tf.constant([2], tf.int64))
