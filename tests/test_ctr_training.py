"""Numeric contracts for optional structure-linked training interventions."""
import importlib
import math

import numpy as np
import pytest


@pytest.fixture(params=["pytorch", "tensorflow"])
def native(request):
    framework = request.param
    module = importlib.import_module(f"model_evo_harness.models.{framework}.ctr_training")
    if framework == "pytorch":
        import torch
        return module, lambda x: torch.tensor(x, dtype=torch.float32), torch
    import tensorflow as tf
    return module, lambda x: tf.constant(x, dtype=tf.float32), tf


def array(value):
    return value.detach().numpy() if hasattr(value, "detach") else value.numpy()


@pytest.mark.parametrize("aggregation", ["logit_mean", "probability_mean"])
def test_consistency_matches_declared_bce_and_detaches_teacher(native, aggregation):
    m, tensor, lib = native
    objective = m.EnsembleConsistency(weight=0.5, aggregation=aggregation)
    logits = tensor([[0., 2.], [-1., 1.]])
    labels = tensor([1., 0.])
    result = objective(logits, labels)
    p = 1 / (1 + np.exp(-array(logits)))
    target = (1 / (1 + np.exp(-array(logits).mean(1))) if aggregation == "logit_mean"
              else p.mean(1))
    bce = lambda p, y: -(y * np.log(p) + (1-y) * np.log1p(-p))
    expected_main = bce(target, array(labels)).mean()
    expected_aux = bce(p, target[:, None]).mean(0).sum()
    np.testing.assert_allclose(array(result["loss"]), expected_main + .5*expected_aux, rtol=1e-6)
    np.testing.assert_allclose(array(result["teacher"]), target, rtol=1e-6)
    if hasattr(lib, "autograd"):
        logits.requires_grad_()
        result = objective(logits, labels)
        assert not result["teacher"].requires_grad
        result["loss"].backward()
        assert lib.isfinite(logits.grad).all()
    else:
        with lib.GradientTape(persistent=True) as tape:
            tape.watch(logits)
            result = objective(logits, labels)
            teacher_sum = lib.reduce_sum(result["teacher"])
        assert tape.gradient(teacher_sum, logits) is None
        assert np.isfinite(array(tape.gradient(result["loss"], logits))).all()


def test_consistency_extreme_logits_and_invalid_labels(native):
    m, tensor, lib = native
    for aggregation in ("logit_mean", "probability_mean"):
        result = m.EnsembleConsistency(aggregation=aggregation)(tensor([[1000., -1000.]]), tensor([1.]))
        assert np.isfinite(array(result["loss"]))
    invalid_input = ValueError if hasattr(lib, "autograd") else lib.errors.InvalidArgumentError
    with pytest.raises(invalid_input):
        m.EnsembleConsistency()(tensor([[0., 1.]]), tensor([.5]))
    with pytest.raises(ValueError):
        m.EnsembleConsistency(weight=-1)


def test_transition_loss_masks_padding_and_has_scalar_zero_empty_case(native):
    m, tensor, lib = native
    objective = m.NextBehaviorObjective(2, 2, 3)
    states = tensor([[[1., 0.], [float("nan"), float("nan")]]])
    positive = tensor([[[0., 1.], [float("nan"), float("nan")]]])
    negative = tensor([[[1., 1.], [float("nan"), float("nan")]]])
    mask = lib.tensor([[True, False]]) if hasattr(lib, "autograd") else lib.constant([[True, False]])
    result = objective(states, positive, negative, mask)
    expected = (np.logaddexp(0, -array(result["positive_logit"])[0, 0]) +
                np.logaddexp(0, array(result["negative_logit"])[0, 0])) / 2
    assert array(result["loss"]).shape == ()
    np.testing.assert_allclose(array(result["loss"]), expected, rtol=1e-6)
    assert int(array(result["valid_count"])) == 1
    empty = objective(states[:, :0], positive[:, :0], negative[:, :0], mask[:, :0])
    assert float(array(empty["loss"])) == 0
    if hasattr(lib, "autograd"):
        result["loss"].backward()
        assert all(p.grad is not None and lib.isfinite(p.grad).all() for p in objective.parameters())
    else:
        with lib.GradientTape() as tape:
            loss = objective(states, positive, negative, mask)["loss"]
        assert all(g is not None and np.isfinite(array(g)).all()
                   for g in tape.gradient(loss, objective.trainable_variables))


def test_coarse_fine_and_parameter_groups_are_independent(native):
    m, tensor, lib = native
    result = m.CoarseFineObjective(coarse_weight=2., fine_weight=.5)(
        tensor([0., 0.]), tensor([0., 0.]), tensor([0., 1.]))
    np.testing.assert_allclose(array(result["loss"]), 2.5*math.log(2), rtol=1e-6)
    a, b = tensor([1., 2.]), tensor([3.])
    regularizer = m.GroupedRegularization(embedding_weight=.2, network_weight=.4)
    result = regularizer([a], [b])
    np.testing.assert_allclose(array(result["loss"]), .2/2*5+.4/2*9, rtol=1e-6)
    with pytest.raises(ValueError, match="overlap|duplicate"):
        regularizer([a], [a])


def test_tensorflow_objectives_execute_in_graph_mode():
    import tensorflow as tf
    m = importlib.import_module("model_evo_harness.models.tensorflow.ctr_training")
    layer = m.NextBehaviorObjective(2, 2, 3)
    @tf.function(input_signature=[tf.TensorSpec([None, None, 2], tf.float32),
                                 tf.TensorSpec([None, None], tf.bool)])
    def run(x, mask):
        return layer(x, x, x, mask)["loss"]
    assert np.isfinite(run(tf.ones([2, 3, 2]), tf.ones([2, 3], tf.bool)).numpy())
    assert run(tf.zeros([2, 0, 2]), tf.zeros([2, 0], tf.bool)).numpy() == 0


def test_finite_extreme_logits_do_not_overflow_before_mean(native):
    m, tensor, lib = native
    result = m.EnsembleConsistency()(tensor([[3e38, 3e38], [3e38, 3e38]]), tensor([1, 1]))
    assert float(array(result['loss'])) == 0
    # Mean branch BCE is representable even when its unreduced sum is not.
    result = m.CoarseFineObjective(.5, .5)(tensor([-3e38, -3e38]), tensor([-3e38, -3e38]), tensor([1, 1]))
    np.testing.assert_allclose(array(result['loss']), 3e38, rtol=1e-6)


def test_zero_weight_regularization_skips_overflowing_powers_and_has_zero_gradients(native):
    m, tensor, lib = native
    if hasattr(lib, 'autograd'):
        a = tensor([3e38, 3e38]).requires_grad_()
        b = tensor([1e30]).requires_grad_()
        result = m.GroupedRegularization()([a], [b])
        result['loss'].backward()
        gradients = [a.grad, b.grad]
    else:
        a, b = lib.Variable([3e38, 3e38]), lib.Variable([1e30])
        with lib.GradientTape() as tape:
            result = m.GroupedRegularization()([a], [b])
        gradients = tape.gradient(result['loss'], [a, b])
    assert float(array(result['loss'])) == 0
    assert all(g is not None and np.all(array(g) == 0) for g in gradients)


def test_zero_consistency_weight_cannot_poison_supervised_loss(native):
    m, tensor, _ = native
    result = m.EnsembleConsistency(weight=0)(tensor([[3e38, -3e38, 3e38, -3e38]]), tensor([1]))
    np.testing.assert_allclose(array(result['loss']), math.log(2), rtol=1e-6)


def test_transition_bce_scales_before_adding_extreme_terms(native):
    m, tensor, lib = native
    objective = m.NextBehaviorObjective(1, 1, 2)
    if hasattr(lib, 'autograd'):
        valid = lib.tensor([[True]])
        with lib.no_grad():
            objective.scorer[0].weight.copy_(tensor([[0, 1], [0, -1]]))
            objective.scorer[0].bias.zero_()
            objective.scorer[2].weight.copy_(tensor([[-3e38, 3e38]]))
            objective.scorer[2].bias.zero_()
    else:
        valid = lib.constant([[True]])
        objective.scorer.layers[0].kernel.assign([[0, 0], [1, -1]])
        objective.scorer.layers[0].bias.assign([0, 0])
        objective.scorer.layers[1].kernel.assign([[-3e38], [3e38]])
        objective.scorer.layers[1].bias.assign([0])
    result = objective(tensor([[[0]]]), tensor([[[1]]]), tensor([[[-1]]]), valid)
    np.testing.assert_allclose(array(result['loss']), 3e38, rtol=1e-6)


@pytest.mark.parametrize('aggregation', ['logit_mean', 'probability_mean'])
def test_consistency_gradient_matches_detached_target_formula(native, aggregation):
    m, tensor, lib = native
    values = np.array([[.3, -1.2], [2., .7]], np.float32)
    targets = np.array([1., 0.], np.float32)
    x = tensor(values)
    if hasattr(lib, 'autograd'):
        x.requires_grad_()
        m.EnsembleConsistency(.4, aggregation)(x, tensor(targets))['loss'].backward()
        gradient = array(x.grad)
    else:
        x = lib.Variable(x)
        with lib.GradientTape() as tape:
            loss = m.EnsembleConsistency(.4, aggregation)(x, tensor(targets))['loss']
        gradient = array(tape.gradient(loss, x))
    probabilities = 1 / (1 + np.exp(-values))
    if aggregation == 'logit_mean':
        teacher = 1 / (1 + np.exp(-values.mean(-1)))
        supervised = (teacher - targets)[:, None] / 4
    else:
        teacher = probabilities.mean(-1)
        supervised = ((teacher - targets) / (teacher * (1-teacher)))[:, None] * probabilities * (1-probabilities) / 4
    expected = supervised + .4 * (probabilities - teacher[:, None]) / 2
    np.testing.assert_allclose(gradient, expected, atol=1e-7, rtol=1e-6)


def test_mapped_transition_scorers_agree_and_padding_has_zero_gradient():
    import torch
    import tensorflow as tf
    pt = importlib.import_module('model_evo_harness.models.pytorch.ctr_training')
    kt = importlib.import_module('model_evo_harness.models.tensorflow.ctr_training')
    source, target = pt.NextBehaviorObjective(2, 2, 3), kt.NextBehaviorObjective(2, 2, 3)
    rng = np.random.default_rng(23)
    values = [rng.normal(size=(2, 3, 2)).astype('float32') for _ in range(3)]
    mask = np.array([[True, False, True], [False, False, False]])
    for x in values: x[~mask] = np.nan
    for left, right in zip((source.scorer[0], source.scorer[2]), target.scorer.layers):
        right.kernel.assign(left.weight.detach().numpy().T)
        right.bias.assign(left.bias.detach().numpy())
    tx, kx = [torch.tensor(x, requires_grad=True) for x in values], [tf.Variable(x) for x in values]
    tresult = source(*tx, torch.tensor(mask))
    tresult['loss'].backward()
    with tf.GradientTape() as tape:
        kresult = target(*kx, tf.constant(mask))
    kgradients = tape.gradient(kresult['loss'], kx + target.trainable_variables)
    for key in ('loss', 'positive_logit', 'negative_logit', 'valid_count'):
        np.testing.assert_allclose(array(tresult[key]), array(kresult[key]), rtol=1e-6, atol=1e-7)
    for left, right in zip(tx, kgradients):
        np.testing.assert_allclose(array(left.grad), array(right), rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(array(left.grad)[~mask], 0)
    assert all(g is not None and np.isfinite(array(g)).all() for g in kgradients)
