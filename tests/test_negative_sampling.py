"""Conditional sampling probabilities, exclusions and local RNG contracts."""
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(params=['pytorch', 'tensorflow'])
def framework(request):
    return request.param


def sampler(framework, **kwargs):
    path = Path(__file__).resolve().parents[1] / 'src/model_evo_harness/models' / framework / 'negative_sampling.py'
    assert path.is_file(), f'Missing native ConditionalNegativeSampler: {path}'
    return importlib.import_module(f'model_evo_harness.models.{framework}.negative_sampling').ConditionalNegativeSampler(**kwargs)


def tensor(framework, value):
    if framework == 'pytorch':
        import torch
        return torch.tensor(np.asarray(value))
    import tensorflow as tf
    return tf.constant(value)


def array(value):
    return value.detach().numpy() if hasattr(value, 'detach') else value.numpy()


def draw(framework, model, counts, eligible, positives, num_samples=1000, seed=19):
    return model(*(tensor(framework, v) for v in (counts, eligible, positives)),
                 num_samples=num_samples, seed=seed)


@pytest.mark.parametrize('settings, expected', [
    ({}, [[0, .5, .5, 0], [0, 0, .5, .5]]),
    ({'distribution': 'popularity', 'alpha': .5}, [[0, .2, .8, 0], [0, 0, 2/3, 1/3]]),
    ({'distribution': 'popularity', 'alpha': .5, 'uniform_mix': .5},
     [[0, .35, .65, 0], [0, 0, 13/22, 9/22]]),
])
def test_returns_actual_conditional_probability_for_each_sample(framework, settings, expected):
    # Counts are intentionally neither sorted nor eligible in the same order.
    model = sampler(framework, **settings)
    counts = [9., 1., 16., 4.]
    eligible = [[True, True, True, False], [False, True, True, True]]
    positives = [[True, False, False, False], [False, True, False, False]]
    result = draw(framework, model, counts, eligible, positives)
    ids, probs = array(result['item_ids']), array(result['sampling_probs'])
    assert ids.shape == probs.shape == (2, 1000)
    expected = np.asarray(expected)
    assert all(set(row) == set(np.flatnonzero(expected[b])) for b, row in enumerate(ids))
    np.testing.assert_allclose(probs, np.take_along_axis(expected, ids, axis=1), rtol=1e-6)


def test_draw_frequencies_follow_returned_nonuniform_distribution(framework):
    model = sampler(framework, distribution='popularity', alpha=1)
    result = draw(framework, model, [1., 9., 90.], [[True]*3],
                  [[False, False, True]], num_samples=40000)
    ids = array(result['item_ids'])[0]
    np.testing.assert_allclose(np.bincount(ids, minlength=3)/len(ids), [.1, .9, 0], atol=.008)


def test_samples_with_replacement_without_mutating_host_inputs(framework):
    model = sampler(framework, distribution='popularity')
    values = [tensor(framework, v) for v in ([1., 8., 2.], [[True]*3], [[True, False, True]])]
    snapshots = [array(v).copy() for v in values]
    result = model.sample(*values, num_samples=5, seed=2)
    np.testing.assert_array_equal(array(result['item_ids']), [[1]*5])
    np.testing.assert_array_equal(array(result['sampling_probs']), [[1.]*5])
    for value, original in zip(values, snapshots):
        np.testing.assert_array_equal(array(value), original)


def test_seed_reproduces_samples_without_changing_global_random_stream(framework):
    model = sampler(framework)
    if framework == 'pytorch':
        import torch
        state = torch.random.get_rng_state().clone()
    else:
        import tensorflow as tf
        tf.random.set_seed(731)
        expected = tf.random.uniform([12])
        tf.random.set_seed(731)
    args = (framework, model, [1.]*5, [[True]*5], [[False]*5])
    first, second = draw(*args, seed=71), draw(*args, seed=71)
    third = draw(*args, seed=72)
    np.testing.assert_array_equal(array(first['item_ids']), array(second['item_ids']))
    assert not np.array_equal(array(first['item_ids']), array(third['item_ids']))
    if framework == 'pytorch':
        np.testing.assert_array_equal(torch.random.get_rng_state(), state)
    else:
        np.testing.assert_array_equal(tf.random.uniform([12]), expected)


@pytest.mark.parametrize('counts, eligible, positives, settings', [
    ([1., 2.], [[False, False]], [[False, False]], {}),
    ([1., 2.], [[True, True]], [[True, True]], {}),
    ([0., 2.], [[True, False]], [[False, False]], {'distribution': 'popularity'}),
    ([0., 0.], [[True, True]], [[False, False]], {'distribution': 'popularity'}),
    ([0., 0.], [[True, True]], [[False, False]], {'distribution': 'popularity', 'uniform_mix': .5}),
])
def test_rejects_zero_support_without_falling_back_to_uniform(framework, counts, eligible, positives, settings):
    model = sampler(framework, **settings)
    error = ValueError
    if framework == 'tensorflow':
        import tensorflow as tf
        error = (ValueError, tf.errors.InvalidArgumentError)
    with pytest.raises(error):
        draw(framework, model, counts, eligible, positives)


@pytest.mark.parametrize('settings', [{}, {'distribution': 'popularity', 'uniform_mix': 1.}])
def test_uniform_distribution_does_not_need_positive_counts(framework, settings):
    result = draw(framework, sampler(framework, **settings), [0., 0.], [[True, True]], [[False, False]])
    np.testing.assert_allclose(array(result['sampling_probs']), .5)


def test_uniform_mixture_gives_zero_count_eligible_items_support(framework):
    model = sampler(framework, distribution='popularity', uniform_mix=.25)
    result = draw(framework, model, [0., 5.], [[True, False]], [[False, False]], num_samples=3)
    np.testing.assert_array_equal(array(result['item_ids']), [[0]*3])
    np.testing.assert_array_equal(array(result['sampling_probs']), [[1.]*3])


@pytest.mark.parametrize('dtype, high, low', [(np.float32, 1e30, 1e-30), (np.float64, 1e300, 1e-300)])
@pytest.mark.parametrize('alpha', [.75, 1e308])
def test_conditioning_preserves_tiny_positive_count_support(framework, dtype, high, low, alpha):
    model = sampler(framework, distribution='popularity', alpha=alpha)
    counts = np.array([high, low, low], dtype=dtype)
    result = draw(framework, model, counts,
                  [[False, True, False], [False, True, True]],
                  [[False]*3, [False]*3])
    ids, probabilities = array(result['item_ids']), array(result['sampling_probs'])
    np.testing.assert_array_equal(ids[0], 1)
    assert set(ids[1]) == {1, 2}
    np.testing.assert_allclose(probabilities[0], 1.)
    np.testing.assert_allclose(probabilities[1], .5)


@pytest.mark.parametrize('dtype, high, low', [(np.float32, 1e30, 1e-30), (np.float64, 1e300, 1e-300)])
def test_extreme_catalog_mixture_retains_popularity_mass_after_exclusion(framework, dtype, high, low):
    # Tail popularity masses are low and 2*low; catalog uniform adds low each.
    # After excluding the head, the two remaining weights are in ratio 2:3.
    model = sampler(framework, distribution='popularity', alpha=.5, uniform_mix=3*low)
    counts = np.array([high, low, 4*low], dtype=dtype)
    result = draw(framework, model, counts, [[False, True, True]], [[False]*3])
    ids, probabilities = array(result['item_ids']), array(result['sampling_probs'])
    assert set(ids[0]) == {1, 2}
    np.testing.assert_allclose(probabilities, np.take([0, .4, .6], ids), rtol=1e-6)


def test_large_finite_alpha_keeps_catalog_ties_and_uniform_mixture_finite(framework):
    model = sampler(framework, distribution='popularity', alpha=1e308, uniform_mix=.25)
    result = draw(framework, model, np.array([2., 2., 1.], np.float32), [[True]*3], [[False]*3])
    ids, probabilities = array(result['item_ids']), array(result['sampling_probs'])
    assert set(ids[0]) == {0, 1, 2}
    np.testing.assert_allclose(probabilities, np.take([11/24, 11/24, 1/12], ids), rtol=1e-6)


def test_small_alpha_retains_relative_tail_probabilities(framework):
    model = sampler(framework, distribution='popularity', alpha=.01)
    counts = np.array([1e30, 1e-30, 1e-29], np.float32)
    result = draw(framework, model, counts, [[False, True, True]], [[False]*3])
    ids, probabilities = array(result['item_ids']), array(result['sampling_probs'])
    assert set(ids[0]) == {1, 2}
    np.testing.assert_allclose(probabilities, np.take([0, .49424379158885506, .5057562084111449], ids), rtol=1e-6)


def test_tiny_positive_mixture_survives_before_conditioning(framework):
    model = sampler(framework, distribution='popularity', uniform_mix=1e-100)
    result = draw(framework, model, np.array([1., 0.], np.float32), [[False, True]], [[False, False]])
    np.testing.assert_array_equal(array(result['item_ids']), 1)
    np.testing.assert_array_equal(array(result['sampling_probs']), 1.)


def test_zero_alpha_assigns_equal_weights_to_zero_and_positive_counts(framework):
    model = sampler(framework, distribution='popularity', alpha=0)
    result = draw(framework, model, np.array([1., 0.], np.float32),
                  [[True, True], [False, True]], [[False, False], [False, False]])
    ids, probabilities = array(result['item_ids']), array(result['sampling_probs'])
    assert set(ids[0]) == {0, 1}
    np.testing.assert_array_equal(ids[1], 1)
    np.testing.assert_allclose(probabilities[0], .5)
    np.testing.assert_allclose(probabilities[1], 1.)


@pytest.mark.parametrize('settings', [
    {'distribution': 'ralm'}, {'alpha': -1}, {'alpha': float('nan')},
    {'alpha': float('inf')}, {'uniform_mix': -.1}, {'uniform_mix': 1.1},
    {'uniform_mix': float('nan')}, {'alpha': True}, {'uniform_mix': False},
    {'alpha': 1j}, {'uniform_mix': '0.5'},
])
def test_rejects_undefined_distribution_parameters(framework, settings):
    with pytest.raises(ValueError):
        sampler(framework, **settings)


@pytest.mark.parametrize('counts, eligible, positives, num_samples', [
    ([-1., 2.], [[True, True]], [[False, False]], 2),
    ([float('nan'), 2.], [[True, True]], [[False, False]], 2),
    ([float('inf'), 2.], [[True, True]], [[False, False]], 2),
    ([1., 2.], [[True, True]], [[False, False]], 0),
    ([1., 2.], [[True, True]], [[False, False]], 1.5),
    ([1., 2.], [[True]], [[False]], 2),
    ([1., 2.], [[True, True]], [[False, False], [False, False]], 2),
    ([1., 2.], [[1, 1]], [[False, False]], 2),
    ([1+2j, 3+0j], [[True, True]], [[False, False]], 2),
    ([True, False], [[True, True]], [[False, False]], 2),
])
def test_rejects_invalid_counts_shapes_masks_and_sample_count(framework, counts, eligible, positives, num_samples):
    model = sampler(framework)
    error = (TypeError, ValueError)
    if framework == 'tensorflow':
        import tensorflow as tf
        error += (tf.errors.InvalidArgumentError,)
    with pytest.raises(error):
        draw(framework, model, counts, eligible, positives, num_samples=num_samples)


@pytest.mark.parametrize('seed', [-1, 2**63, True, 1.5])
def test_rejects_seeds_outside_shared_signed_integer_range(framework, seed):
    model = sampler(framework)
    with pytest.raises(ValueError):
        draw(framework, model, [1., 2.], [[True, True]], [[False, False]], seed=seed)


def test_largest_supported_seed_is_reproducible(framework):
    args = (framework, sampler(framework), [1., 2.], [[True, True]], [[False, False]])
    first = draw(*args, seed=2**63-1)
    second = draw(*args, seed=2**63-1)
    np.testing.assert_array_equal(array(first['item_ids']), array(second['item_ids']))


def test_tensorflow_dynamic_graph_enforces_shape_and_support_contract():
    import tensorflow as tf
    model = sampler('tensorflow', distribution='popularity', alpha=1, uniform_mix=.5)
    @tf.function(autograph=False, input_signature=[tf.TensorSpec([None], tf.float32),
                                                  tf.TensorSpec([None, None], tf.bool),
                                                  tf.TensorSpec([None, None], tf.bool)])
    def run(counts, eligible, positives):
        return model(counts, eligible, positives, num_samples=20, seed=9)
    result = run([1., 3.], [[True, True]], [[False, False]])
    ids = result['item_ids'].numpy()
    np.testing.assert_allclose(result['sampling_probs'], np.take([.375, .625], ids))
    with pytest.raises(tf.errors.InvalidArgumentError):
        run([1., 3.], [[True, True]], [[True, True]])
