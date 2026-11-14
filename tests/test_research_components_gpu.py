"""GPU contracts for research components; no dataset training is performed."""

import importlib
from contextlib import contextmanager

import numpy as np
import pytest


RELATIONS = ("user_history", "target_history", "target_user")


@pytest.fixture(params=["pytorch", "tensorflow"])
def gpu(request):
    framework = request.param
    if framework == "pytorch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU unavailable")
        with torch.cuda.device(0):
            yield framework
    else:
        tf = pytest.importorskip("tensorflow")
        if not tf.config.list_physical_devices("GPU"):
            pytest.skip("TensorFlow GPU unavailable")
        with tf.device("/GPU:0"):
            yield framework


def _module(framework, name, area="models"):
    return importlib.import_module(f"model_evo_harness.{area}.{framework}.{name}")


def _tensor(framework, values, *, tracked=False, dtype="float32"):
    values = np.asarray(values)
    if values.dtype != bool:
        values = values.astype(dtype)
    if framework == "pytorch":
        import torch
        return torch.tensor(values, device="cuda:0", requires_grad=tracked)
    import tensorflow as tf
    return tf.Variable(values) if tracked else tf.constant(values)


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value.numpy()


def _assert_gpu(value):
    assert value is not None
    if hasattr(value, "is_cuda"):
        assert value.is_cuda
    else:
        assert "GPU:0" in value.device
    assert np.isfinite(_array(value)).all()


@contextmanager
def _precision(framework, half):
    if framework == "pytorch":
        import torch
        with torch.autocast("cuda", dtype=torch.float16, enabled=half):
            yield
    else:
        import tensorflow as tf
        policy = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16" if half else "float32")
            yield
        finally:
            tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_representation_metrics_on_gpu_preserve_rank_and_mask_contracts(gpu, dtype):
    module = _module(gpu, "research_diagnostics", "analysis")
    values = np.array([[[3., 0.], [0., 1.], [np.nan, np.inf]],
                       [[np.nan, np.inf]] * 3], np.float32)
    x = _tensor(gpu, values, tracked=True, dtype=dtype)
    mask = _tensor(gpu, [[True, True, False], [False, False, False]])
    result = module.RepresentationDiagnostics()(x, mask, reference=x)
    expected_rank = np.exp(-.75 * np.log(.75) - .25 * np.log(.25))
    expected = {"effective_rank": [expected_rank, 0], "stable_rank": [10 / 9, 0],
                "output_variance": [1.25, 0], "valid_tokens": [2, 0],
                "branch_similarity": [1, 0]}
    for name, value in result.items():
        _assert_gpu(value)
        np.testing.assert_allclose(_array(value), expected[name], atol=2e-6)
        if name != "valid_tokens":
            assert "float32" in str(value.dtype)
        if gpu == "pytorch":
            assert not value.requires_grad
    centered = module.RepresentationDiagnostics(center=True)(x, mask, reference=x)
    np.testing.assert_allclose(_array(centered["effective_rank"]), [1, 0], atol=2e-6)
    empty = _tensor(gpu, np.empty((2, 0, 2)), dtype=dtype)
    for value in module.RepresentationDiagnostics()(empty, reference=empty).values():
        _assert_gpu(value)
        np.testing.assert_array_equal(_array(value), [0, 0])


def test_gradient_screening_on_gpu_uses_objective_gradients_and_protected_groups(gpu):
    module = _module(gpu, "research_diagnostics", "analysis")
    selector = module.GradientFeatureSelector()
    # The last two fields have zero marginal objective gradient but must survive
    # together when the caller declares them an interaction-only group.
    values = [[[1.], [1.], [np.nan], [4.], [5.]],
              [[2.], [np.nan], [np.nan], [4.], [5.]],
              [[np.nan], [3.], [np.nan], [4.], [5.]]]
    valid = np.array([[1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [0, 1, 0, 1, 1]], bool)
    x = _tensor(gpu, values, tracked=True)
    mask = _tensor(gpu, valid)
    coefficients = _tensor(gpu, [[[1.], [1.], [0.], [0.], [0.]]])
    if gpu == "pytorch":
        import torch
        safe = torch.where(mask[..., None], x, 0)
        gradient, = torch.autograd.grad((safe * coefficients).sum(), x)
    else:
        import tensorflow as tf
        with tf.GradientTape() as tape:
            safe = tf.where(mask[..., None], x, 0)
            loss = tf.reduce_sum(safe * coefficients)
        gradient = tape.gradient(loss, x)
    _assert_gpu(gradient)
    np.testing.assert_array_equal(_array(gradient)[~valid], 0)
    result = selector(x, gradient, mask)
    for value in result.values():
        _assert_gpu(value)
    np.testing.assert_allclose(_array(result["saliency"]), [1.5, 2., 0, 0, 0])
    np.testing.assert_allclose(_array(result["redundancy"])[:3, :3],
                               [[1, 1, 0], [1, 1, 0], [0, 0, 0]], atol=2e-6)
    np.testing.assert_array_equal(_array(result["valid_counts"]), valid.sum(0))
    np.testing.assert_array_equal(_array(result["overlap_counts"]),
                                  valid.astype(np.int64).T @ valid.astype(np.int64))
    assert "int64" in str(result["overlap_counts"].dtype)
    assert selector.select(result["saliency"], 3, protected_groups=[(3, 4)]) == (1, 3, 4)
    assert selector.select(result["saliency"], 3, protected_groups=[(2, 3), (3, 4)]) == (2, 3, 4)


def test_progressive_interaction_on_gpu_keeps_initial_masks_and_finite_gradients(gpu):
    module = _module(gpu, "progressive_interaction")
    model = module.ProgressiveInteraction(3, 4, [0, 2, 9], hidden_dim=5)
    valid = np.array([[True, False, True], [False, False, False]])
    x = _tensor(gpu, [[[3., 0., 1., 0.], [np.nan, np.inf, -np.inf, np.nan], [1., 0., 2., 0.]],
                      [[np.nan] * 4] * 3], tracked=True)
    mask = _tensor(gpu, valid)
    coefficients = _tensor(gpu, np.arange(24).reshape(2, 3, 4) / 24)
    if gpu == "pytorch":
        model = model.cuda()
        output, masks = model(x, mask, return_masks=True)
        (output * coefficients).sum().backward()
        gradients = [x.grad] + [parameter.grad for parameter in model.parameters()]
    else:
        import tensorflow as tf
        with tf.GradientTape() as tape:
            output, masks = model(x, mask, return_masks=True)
            loss = tf.reduce_sum(output * coefficients)
        gradients = tape.gradient(loss, [x] + model.trainable_variables)
    for value in [output, masks, *gradients]:
        _assert_gpu(value)
    np.testing.assert_array_equal(_array(output)[~valid], 0)
    np.testing.assert_array_equal(_array(gradients[0])[~valid], 0)
    assert np.abs(_array(gradients[0])[valid]).sum() > 0
    np.testing.assert_array_equal(_array(masks).sum((-2, -1)), [[0, 2, 4], [0, 0, 0]])
    # Affinity ties (0,2) and (2,0) use row-major order from the input.
    np.testing.assert_array_equal(_array(masks)[0, 1], [[1, 0, 1], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(_array(masks), _array(model.interaction_masks(x, mask)))
    assert np.all(_array(masks)[:, :-1] <= _array(masks)[:, 1:])
    clean = _tensor(gpu, np.nan_to_num(_array(x), nan=0, posinf=0, neginf=0))
    np.testing.assert_allclose(_array(output), _array(model(clean, mask)), atol=2e-6)


@pytest.mark.parametrize("half", [False, True], ids=["float32", "mixed_float16"])
def test_context_and_relation_composition_on_gpu_has_masked_finite_gradients(gpu, half):
    module = _module(gpu, "contextual_history")
    rng = np.random.default_rng(71)
    validity = [np.array([[1, 1], [0, 0]], bool),
                np.array([[1, 1, 0], [0, 0, 0]], bool),
                np.array([[1, 1, 0], [1, 0, 0]], bool)]
    arrays = [np.where(mask[..., None], rng.normal(0, .2, (*mask.shape, 4)), np.nan)
              for mask in validity]
    arrays.append(np.where(validity[1][..., None], rng.normal(0, .2, (2, 3, 2)), np.nan))
    values = [_tensor(gpu, value, tracked=True, dtype="float16" if half else "float32")
              for value in arrays]
    masks = [_tensor(gpu, mask) for mask in validity]
    with _precision(gpu, half):
        encoder = module.ContextualHistoryEncoder(4, 2, 4)
        relation = module.RelationHistoryAttention(4, 2)
        if gpu == "pytorch":
            encoder, relation = encoder.cuda(), relation.cuda()
            encoded, hm = encoder(values[1], values[3], masks[1])
            result = relation(values[0], encoded, values[2], masks[0], hm, masks[2], return_attention=True)
            sum(result[name].float().square().sum() for name in RELATIONS).backward()
            gradients = [value.grad for value in values]
            gradients += [parameter.grad for model in (encoder, relation) for parameter in model.parameters()]
        else:
            import tensorflow as tf
            with tf.GradientTape() as tape:
                encoded, hm = encoder(values[1], values[3], masks[1])
                result = relation(values[0], encoded, values[2], masks[0], hm, masks[2], return_attention=True)
                loss = sum(tf.reduce_sum(tf.square(tf.cast(result[name], tf.float32))) for name in RELATIONS)
            gradients = tape.gradient(loss, values + encoder.trainable_variables + relation.trainable_variables)
        _assert_gpu(encoded)
        assert "float16" in str(encoded.dtype) if half else "float32" in str(encoded.dtype)
        np.testing.assert_array_equal(_array(hm), validity[1])
        np.testing.assert_array_equal(_array(encoded)[~validity[1]], 0)
        for gradient in gradients:
            _assert_gpu(gradient)
        for gradient, mask in zip(gradients[:4], (*validity, validity[1])):
            np.testing.assert_array_equal(_array(gradient)[~mask], 0)
            assert np.abs(_array(gradient)[mask]).sum() > 0
        for name in RELATIONS:
            output, weights, valid = result[name], result["attention"][name], result[name + "_mask"]
            for value in (output, weights, valid):
                _assert_gpu(value)
            np.testing.assert_array_equal(_array(output)[~_array(valid)], 0)
            expected = np.broadcast_to(_array(valid)[:, None, :], _array(weights).shape[:-1])
            np.testing.assert_allclose(_array(weights).sum(-1), expected, atol=2e-6)
        # Each candidate reads the same evidence independently of other targets.
        single = relation(values[0], encoded, values[2][:, :1], masks[0], hm, masks[2][:, :1])
        for name in ("target_history", "target_user"):
            np.testing.assert_allclose(_array(single[name]), _array(result[name])[:, :1],
                                       atol=2e-3 if half else 2e-6, rtol=2e-3 if half else 2e-6)


@pytest.mark.parametrize("lengths", [(2, 0, 3), (0, 0, 0)])
def test_context_and_relation_empty_sequences_remain_on_gpu(gpu, lengths):
    module = _module(gpu, "contextual_history")
    encoder, relation = module.ContextualHistoryEncoder(4, 2, 4), module.RelationHistoryAttention(4, 2)
    if gpu == "pytorch":
        encoder, relation = encoder.cuda(), relation.cuda()
    values = [_tensor(gpu, np.ones((2, length, 4))) for length in lengths]
    masks = [_tensor(gpu, np.ones((2, length), bool)) for length in lengths]
    encoded, hm = encoder(values[1], _tensor(gpu, np.ones((2, lengths[1], 2))), masks[1])
    _assert_gpu(encoded)
    assert tuple(encoded.shape) == (2, lengths[1], 4)
    result = relation(values[0], encoded, values[2], masks[0], hm, masks[2], return_attention=True)
    for name, q, k in zip(RELATIONS, (lengths[0], lengths[2], lengths[2]),
                         (lengths[1], lengths[1], lengths[0])):
        output, weights, valid = result[name], result["attention"][name], result[name + "_mask"]
        for value in (output, weights, valid):
            _assert_gpu(value)
        assert tuple(output.shape) == (2, q, 4)
        assert tuple(weights.shape) == (2, 2, q, k)
        if k == 0:
            np.testing.assert_array_equal(_array(output), 0)
            np.testing.assert_array_equal(_array(weights), 0)
            np.testing.assert_array_equal(_array(valid), False)
