"""Numerical contracts for native sequence composition; no third-party DL wrappers."""
import importlib
import os

import numpy as np
import pytest


@pytest.fixture(params=['pytorch', 'tensorflow'])
def backend(request):
    name = request.param
    module = importlib.import_module('torch' if name == 'pytorch' else 'tensorflow')
    required = os.environ.get('MODELEVO_REQUIRE_GPU') == '1'
    available = module.cuda.is_available() if name == 'pytorch' else bool(module.config.list_physical_devices('GPU'))
    if required and not available:
        pytest.fail(f'{name} GPU required')
    device = ('cuda' if available else 'cpu') if name == 'pytorch' else ('/GPU:0' if available else '/CPU:0')
    with module.device(device):
        yield name, module


def load(backend, file, symbol):
    return getattr(importlib.import_module(f'model_evo_harness.models.{backend[0]}.{file}'), symbol)


def tensor(backend, values, boolean=False):
    name, lib = backend
    return lib.tensor(values, dtype=lib.bool if boolean else lib.float32) if name == 'pytorch' else lib.constant(values, dtype=lib.bool if boolean else lib.float32)


def array(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else x.numpy()


def gradients(backend, model, fn):
    name, lib = backend
    if name == 'pytorch':
        output = fn()
        output.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
    else:
        with lib.GradientTape() as tape:
            output = fn()
            total = lib.reduce_sum(output)
        grads = [g for g in tape.gradient(total, model.trainable_variables) if g is not None]
    assert grads and all(np.isfinite(array(g)).all() for g in grads)
    if os.environ.get('MODELEVO_REQUIRE_GPU') == '1':
        assert all(g.is_cuda for g in [output, *grads]) if name == 'pytorch' else all('GPU:0' in g.device.upper() for g in [output, *grads])
    return output, grads


def inputs(backend):
    return (tensor(backend, [[.3, -.4], [.5, .2]]),
            tensor(backend, [[[.1, .2], [.3, -.1], [float('nan'), float('nan')]],
                             [[float('nan'), float('nan')]] * 3]),
            tensor(backend, [[True, True, False], [False, False, False]], True))



def active_objective(backend, hidden_dim=3):
    """Keep the auxiliary ReLU active so gradient assertions are deterministic."""
    objective = load(backend, 'ctr_training', 'NextBehaviorObjective')(2, 2, hidden_dim)
    if backend[0] == 'pytorch':
        with backend[1].no_grad():
            for layer in objective.scorer:
                if hasattr(layer, 'weight'):
                    layer.weight.fill_(.1)
                    layer.bias.fill_(1.)
    else:
        for layer in objective.scorer.layers:
            layer.kernel.assign(backend[1].fill(layer.kernel.shape, .1))
            layer.bias.assign(backend[1].ones_like(layer.bias))
    return objective


def test_din_padding_is_removed_before_forward_and_backward(backend):
    target, history, mask = inputs(backend)
    model = load(backend, 'architectures', 'DIN')(2, 4)
    output, _ = gradients(backend, model, lambda: model(target, history, mask))
    clean = tensor(backend, np.nan_to_num(array(history)))
    np.testing.assert_allclose(array(output), array(model(target, clean, mask)), atol=1e-6)
    invalid = tensor(backend, np.full((2, 3, 2), np.nan))
    with pytest.raises((ValueError, RuntimeError, backend[1].errors.InvalidArgumentError) if backend[0] == 'tensorflow' else (ValueError, RuntimeError)):
        model(target, invalid, mask)


def test_dien_prefix_and_padding_contract(backend):
    target, history, mask = inputs(backend)
    model = load(backend, 'sequence', 'DIEN')(2, 4)
    output, _ = gradients(backend, model, lambda: model(target, history, mask))
    np.testing.assert_allclose(array(output), array(model(target, tensor(backend, np.nan_to_num(array(history))), mask)), atol=1e-6)
    holes = tensor(backend, [[True, False, True], [False, False, False]], True)
    with pytest.raises(Exception, match='prefix'):
        model(target, tensor(backend, np.zeros((2, 3, 2))), holes)


def test_masked_dice_statistics_ignore_padding_and_eval_is_frozen(backend):
    name, lib = backend
    model = load(backend, 'sequence_attention', 'MaskedDice')(2, momentum=0.)
    values = tensor(backend, [[[1., 2.], [3., 6.], [np.nan, np.nan]]])
    valid = tensor(backend, [[True, True, False]], True)
    if name == 'pytorch':
        result = model(values, valid)
        model.eval()
    else:
        result = model(values, valid, training=True)
    expected = np.array([[[1., 2.], [3., 6.], [0., 0.]]])
    expected[:, :2] *= 1 / (1 + np.exp(-np.array([[[-1., -1.], [1., 1.]]]) / np.sqrt(1 + np.array([1e-5, 1e-5 / 4]))))
    np.testing.assert_allclose(array(result), expected, atol=1e-5)
    mean = array(model.moving_mean).copy()
    variance = array(model.moving_variance).copy()
    if name == 'pytorch':
        model(values * 2, valid)
    else:
        model(values * 2, valid, training=False)
    np.testing.assert_array_equal(array(model.moving_mean), mean)
    np.testing.assert_array_equal(array(model.moving_variance), variance)
    empty = tensor(backend, [[False, False, False]], True)
    if name == 'pytorch':
        model.train()
        output = model(values, empty)
    else:
        output = model(values, empty, training=True)
    np.testing.assert_array_equal(array(output), np.zeros((1, 3, 2)))
    np.testing.assert_array_equal(array(model.moving_mean), mean)


@pytest.mark.parametrize('normalization', ['raw', 'sigmoid', 'softmax'])
def test_local_activation_pooling_analytic_weights(backend, normalization):
    name, lib = backend
    pool = load(backend, 'sequence_attention', 'LocalActivationPooling')(2, 3, normalization=normalization)
    target, history, mask = inputs(backend)
    pool(target, history, mask)
    if name == 'pytorch':
        with lib.no_grad():
            for p in pool.parameters():
                p.zero_()
            pool.score.bias.fill_(2.)
    else:
        for p in pool.trainable_variables:
            p.assign(lib.zeros_like(p))
        pool.score.bias.assign([2.])
    result = pool(target, history, mask)
    weight = {'raw': 2., 'sigmoid': 1/(1+np.exp(-2)), 'softmax': .5}[normalization]
    np.testing.assert_allclose(array(result['attention']), [[weight, weight, 0.], [0., 0., 0.]], atol=1e-6)
    np.testing.assert_allclose(array(result['representation']), [[.4*weight, .1*weight], [0., 0.]], atol=1e-6)
    gradients(backend, pool, lambda: pool(target, history, mask)['representation'])


def test_din_accepts_reusable_dice_pooling(backend):
    target, history, mask = inputs(backend)
    pool = load(backend, 'sequence_attention', 'LocalActivationPooling')(2, 4, activation='dice', normalization='softmax')
    model = load(backend, 'architectures', 'DIN')(2, 4, pooling=pool)
    gradients(backend, model, lambda: model(target, history, mask))


@pytest.mark.parametrize('evolution', ['gru', 'aigru', 'agru', 'augru'])
def test_dien_evolution_and_auxiliary_shift(backend, evolution):
    target, history, mask = inputs(backend)
    objective = active_objective(backend)
    model = load(backend, 'sequence', 'DIEN')(2, 4, evolution=evolution, auxiliary_objective=objective)
    details = model(target, history, mask, return_details=True)
    assert array(details['interests']).shape == (2, 3, 2)
    np.testing.assert_array_equal(array(details['interests'])[1], np.zeros((3, 2)))
    negative = tensor(backend, np.nan_to_num(array(history)) + .4)
    result = model.auxiliary_next_event_loss(history, mask, negative, interests=details['interests'])
    manual = objective(details['interests'][:, :-1], history[:, 1:], negative[:, 1:], mask[:, :-1] & mask[:, 1:])
    np.testing.assert_allclose(array(result['loss']), array(manual['loss']), atol=1e-7)
    assert int(array(result['valid_count'])) == 1
    _, grads = gradients(backend, model, lambda: model.auxiliary_next_event_loss(history, mask, negative)['loss'])
    assert any(np.abs(array(g)).sum() > 0 for g in grads)
    empty = tensor(backend, np.zeros((2, 3), dtype=bool), True)
    assert float(array(model.auxiliary_next_event_loss(history, empty, negative)['loss'])) == 0.


def test_dsin_bias_is_opt_in_and_trainable(backend):
    target = tensor(backend, [[.3, -.4]])
    sessions = tensor(backend, [[[[.1, .4], [.8, -.6]], [[.4, .1], [-.2, .7]]]])
    mask = tensor(backend, [[[True, True], [True, True]]], True)
    model = load(backend, 'sequence', 'DSIN')(2, 4, 1, bias_encoding=True, max_sessions=2, max_session_len=2)
    gradients(backend, model, lambda: model(target, sessions, mask))
    if backend[0] == 'pytorch':
        assert model.position_bias.grad is not None and model.session_bias.grad is not None
    else:
        assert any('position_bias' in v.name for v in model.trainable_variables)


def test_bst_history_only_states_do_not_depend_on_candidate(backend):
    pool = load(backend, 'sequence_attention', 'LocalActivationPooling')(2, 3, normalization='softmax')
    model = load(backend, 'history_matching', 'BSTHistoryEncoder')(2, 1, 4, readout='history_attention', pooling=pool)
    target, history, mask = inputs(backend)
    first = model(target, history, mask)
    second = model(-target, history, mask)
    np.testing.assert_allclose(array(first['states']), array(second['states']), atol=1e-6)
    np.testing.assert_array_equal(array(first['representation'])[1], [0., 0.])
    gradients(backend, model, lambda: model(target, history, mask)['representation'])


@pytest.mark.parametrize('length', [0, 1])
def test_empty_or_single_event_has_zero_auxiliary_loss(backend, length):
    objective = active_objective(backend)
    model = load(backend, 'sequence', 'DIEN')(2, 4, auxiliary_objective=objective)
    history = tensor(backend, np.zeros((2, length, 2)))
    mask = tensor(backend, np.ones((2, length), bool), True)
    target = tensor(backend, np.ones((2, 2)))
    result = model(target, history, mask, return_details=True)
    assert array(result['interests']).shape == (2, length, 2)
    loss, _ = gradients(backend, model, lambda: model.auxiliary_next_event_loss(history, mask, history)['loss'])
    assert float(array(loss)) == 0.
    if length == 0:
        np.testing.assert_array_equal(array(result['representation']), np.zeros((2, 2)))
    pool = load(backend, 'sequence_attention', 'LocalActivationPooling')(2, 4, activation='dice', normalization='softmax')
    pooled = pool(target, history, mask)
    if length == 0:
        np.testing.assert_array_equal(array(pooled['representation']), np.zeros((2, 2)))


@pytest.mark.parametrize('mode', ['gru', 'aigru', 'agru', 'augru'])
def test_evolution_cell_matches_same_analytic_formula_in_both_frameworks(backend, mode):
    cell = load(backend, 'sequence', '_AUGRUCell')(2, mode)
    x = tensor(backend, [[.2, -.3]])
    previous = tensor(backend, [[.4, -.1]])
    attention = tensor(backend, [.25])
    cell(x, previous, attention)
    if backend[0] == 'pytorch':
        with backend[1].no_grad():
            cell.gates.weight.fill_(.1); cell.gates.bias.zero_()
            cell.candidate.weight.fill_(.2); cell.candidate.bias.zero_()
    else:
        cell.gates.kernel.assign(backend[1].fill(cell.gates.kernel.shape, .1))
        cell.gates.bias.assign(backend[1].zeros_like(cell.gates.bias))
        cell.candidate.kernel.assign(backend[1].fill(cell.candidate.kernel.shape, .2))
        cell.candidate.bias.assign(backend[1].zeros_like(cell.candidate.bias))
    xx = np.array([[.2, -.3]]) * (.25 if mode == 'aigru' else 1.)
    hh = np.array([[.4, -.1]])
    reset = 1/(1+np.exp(-.1*np.sum(xx+hh)))
    proposal = np.tanh(.2*np.sum(xx+reset*hh))
    update = .25 if mode == 'agru' else reset * (.25 if mode == 'augru' else 1.)
    result, _ = gradients(backend, cell, lambda: cell(x, previous, attention))
    np.testing.assert_allclose(array(result), hh+update*(proposal-hh), atol=1e-6)


def test_dien_auxiliary_loss_reaches_extractor(backend):
    objective = active_objective(backend, 4)
    model = load(backend, 'sequence', 'DIEN')(2, 4, auxiliary_objective=objective)
    target, history, mask = inputs(backend)
    negative = tensor(backend, np.nan_to_num(array(history)) - .7)
    if backend[0] == 'pytorch':
        loss = model.auxiliary_next_event_loss(history, mask, negative)['loss']
        loss.backward()
        grads = [p.grad for p in model.extractor.parameters()]
    else:
        with backend[1].GradientTape() as tape:
            loss = model.auxiliary_next_event_loss(history, mask, negative)['loss']
        grads = tape.gradient(loss, model.extractor.trainable_variables)
    assert all(g is not None and np.isfinite(array(g)).all() for g in grads)
    assert sum(float(np.abs(array(g)).sum()) for g in grads) > 0


def test_din_default_matches_original_sigmoid_formula(backend):
    model = load(backend, 'architectures', 'DIN')(2, 4)
    target, history, mask = inputs(backend)
    history = tensor(backend, np.nan_to_num(array(history)))
    actual = model(target, history, mask)
    if backend[0] == 'pytorch':
        lib = backend[1]
        repeated = target[:, None].expand_as(history)
        joined = lib.cat([repeated, history, repeated-history, repeated*history], -1)
        weights = model.attention(joined).squeeze(-1).sigmoid() * mask
        expected = model.head(lib.cat([target, (history*weights[..., None]).sum(1)], -1)).squeeze(-1)
    else:
        lib = backend[1]
        repeated = lib.broadcast_to(target[:, None], lib.shape(history))
        joined = lib.concat([repeated, history, repeated-history, repeated*history], -1)
        weights = lib.sigmoid(lib.squeeze(model.attention(joined), -1)) * lib.cast(mask, lib.float32)
        expected = lib.squeeze(model.head(lib.concat([target, lib.reduce_sum(history*weights[..., None], 1)], -1)), -1)
    np.testing.assert_allclose(array(actual), array(expected), atol=1e-6)


def test_tensorflow_dien_and_dice_dynamic_shapes():
    import tensorflow as tf
    from model_evo_harness.models.tensorflow.sequence import DIEN
    from model_evo_harness.models.tensorflow.sequence_attention import MaskedDice
    gpu = bool(tf.config.list_physical_devices('GPU'))
    if os.environ.get('MODELEVO_REQUIRE_GPU') == '1' and not gpu:
        pytest.fail('tensorflow GPU required')
    with tf.device('/GPU:0' if gpu else '/CPU:0'):
        model = DIEN(2, 4)
        dice = MaskedDice(2)
        @tf.function(input_signature=[tf.TensorSpec([None, None, 2], tf.float32), tf.TensorSpec([None, None], tf.bool)])
        def run(history, mask):
            return model(tf.ones([tf.shape(history)[0], 2]), history, mask), dice(history, mask, training=True)
        for length in (0, 1, 3):
            logits, values = run(tf.ones([2, length, 2]), tf.ones([2, length], tf.bool))
            assert np.isfinite(array(logits)).all() and np.isfinite(array(values)).all()


def test_dsin_empty_sessions_do_not_change_bidirectional_evolution(backend):
    name, lib = backend
    model = load(backend, 'sequence', 'DSIN')(2, 4, 1, bias_encoding=True,
                                              max_sessions=4, max_session_len=2)
    target = tensor(backend, [[.3, -.4], [.5, .2]])
    data = np.array([[[[.1, .4], [.8, -.6]], [[.4, .1], [-.2, .7]]],
                     [[[np.nan, np.nan], [np.nan, np.nan]]] * 2], dtype=np.float32)
    mask_data = np.array([[[True, True], [True, True]], [[False, False], [False, False]]])
    sessions = tensor(backend, data)
    mask = tensor(backend, mask_data, True)
    model(target, sessions, mask)
    # Remove explicit index biases for the hole/compact equivalence fixture.
    # Padding invariance itself must also hold with learned nonzero biases.
    padded = tensor(backend, np.concatenate([data, np.full((2, 2, 2, 2), np.nan)], 1))
    padded_mask = tensor(backend, np.concatenate([mask_data, np.zeros((2, 2, 2), bool)], 1), True)
    expected = model(target, sessions, mask)
    actual, _ = gradients(backend, model, lambda: model(target, padded, padded_mask))
    np.testing.assert_allclose(array(actual), array(expected), atol=1e-6)
    if name == 'pytorch':
        with lib.no_grad():
            model.session_bias.zero_()
    else:
        model.session_bias.assign(lib.zeros_like(model.session_bias))
    with_hole = tensor(backend, np.concatenate([data[:, :1], np.full((2, 1, 2, 2), np.nan), data[:, 1:]], 1))
    hole_mask = tensor(backend, np.concatenate([mask_data[:, :1], np.zeros((2, 1, 2), bool), mask_data[:, 1:]], 1), True)
    np.testing.assert_allclose(array(model(target, with_hole, hole_mask)),
                               array(model(target, sessions, mask)), atol=1e-6)
