"""Native CUDA regressions for training references; no dataset training occurs."""

import importlib
import os

import numpy as np
import pytest


@pytest.fixture(params=["pytorch", "tensorflow"])
def gpu(request):
    framework = request.param
    required = os.environ.get("MODELEVO_REQUIRE_GPU") == "1"
    try:
        library = importlib.import_module("torch" if framework == "pytorch" else "tensorflow")
    except ImportError:
        if required:
            pytest.fail(f"{framework} is required for this GPU run")
        pytest.skip(f"{framework} unavailable")
    available = (library.cuda.is_available() if framework == "pytorch"
                 else bool(library.config.list_physical_devices("GPU")))
    if not available:
        if required:
            pytest.fail(f"{framework} CUDA GPU is required for this run")
        pytest.skip(f"{framework} CUDA GPU unavailable")
    device = library.cuda.device(0) if framework == "pytorch" else library.device("/GPU:0")
    with device:
        yield framework


def _class(framework, module, name):
    return getattr(importlib.import_module(f"model_evo_harness.models.{framework}.{module}"), name)


def _tensor(framework, values, dtype="float32"):
    values = np.asarray(values, dtype=dtype)
    if framework == "pytorch":
        import torch
        return torch.tensor(values, device="cuda:0")
    import tensorflow as tf
    return tf.constant(values)


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value.numpy()


def _assert_gpu(value):
    assert value is not None
    if hasattr(value, "is_cuda"):
        assert value.is_cuda
    else:
        # Keras 3 variables expose their resource handle instead of .device.
        device = value.device if hasattr(value, "device") else value.handle.device
        assert "GPU:0" in device
    assert np.isfinite(_array(value)).all()


def _assign(framework, parameter, values):
    if framework == "pytorch":
        import torch
        with torch.no_grad():
            parameter.copy_(_tensor(framework, values))
    else:
        parameter.assign(values)


def _sum_gradient(framework, layer, ids):
    if framework == "pytorch":
        layer.zero_grad(set_to_none=True)
        output = layer(ids)
        output.sum().backward()
        gradient = layer.weight.grad
    else:
        import tensorflow as tf
        with tf.GradientTape() as tape:
            output = layer(ids)
            loss = tf.reduce_sum(output)
        gradient = tf.convert_to_tensor(tape.gradient(loss, layer.weight))
    _assert_gpu(output)
    _assert_gpu(gradient)
    return gradient


@pytest.mark.parametrize("kind,threshold_gradient", [
    ("global", [[-.55]]),
    ("dimension", [[-.35, -.2, 0.]]),
    ("feature", [[-.35], [-.2], [0.]]),
    ("feature_dim", [[-.15, -.2, 0.], [-.2, 0., 0.], [0., 0., 0.]]),
])
def test_pruning_search_gpu_gradients_and_optimizer_follow_soft_threshold(gpu, kind, threshold_gradient):
    cls = _class(gpu, "embedding_pruning", "PrunableEmbedding")
    initial = [[.8, -.9, .2], [-.7, .5, -.4], [9., 9., 9.]]
    layer = cls(_tensor(gpu, initial), threshold_type=kind, threshold_init=0., padding_idx=2)
    ids = _tensor(gpu, [[0, 1], [1, 2]], "int64")
    if gpu == "pytorch":
        import torch
        output = layer(ids)
        output.square().sum().backward()
        gradients = [layer.weight.grad, layer.threshold_logits.grad]
        optimizer = torch.optim.SGD(layer.parameters(), lr=.1)
    else:
        import tensorflow as tf
        with tf.GradientTape() as tape:
            output = layer(ids)
            loss = tf.reduce_sum(output ** 2)
        gradients = tape.gradient(loss, [layer.weight, layer.threshold_logits])
        gradients = [tf.convert_to_tensor(gradient) for gradient in gradients]
        optimizer = tf.keras.optimizers.SGD(.1)
    for value in [output, layer.weight, layer.threshold_logits, *gradients]:
        _assert_gpu(value)
    np.testing.assert_allclose(_array(output),
        [[[.3, -.4, 0.], [-.2, 0., 0.]], [[-.2, 0., 0.], [0., 0., 0.]]], atol=1e-6)
    np.testing.assert_allclose(_array(gradients[0]),
        [[.6, -.8, 0.], [-.8, 0., 0.], [0., 0., 0.]], atol=1e-6)
    np.testing.assert_allclose(_array(gradients[1]), threshold_gradient, atol=1e-6)
    if gpu == "pytorch":
        optimizer.step()
    else:
        optimizer.apply_gradients(zip(gradients, [layer.weight, layer.threshold_logits]))
    np.testing.assert_allclose(_array(layer.weight),
        [[.74, -.82, .2], [-.62, .5, -.4], [0., 0., 0.]], atol=1e-6)
    np.testing.assert_allclose(_array(layer.threshold_logits), -.1 * np.asarray(threshold_gradient), atol=1e-6)
    _assert_gpu(layer(ids))


def test_pruning_rewind_gpu_preserves_fixed_support_and_trains_active_zero(gpu):
    cls = _class(gpu, "embedding_pruning", "PrunableEmbedding")
    initial = _tensor(gpu, [[.6, -.8, .2], [.4, 0., .1], [9., 9., 9.]])
    search = cls(initial, threshold_init=0., padding_idx=2)
    _assign(gpu, search.weight, [[.7, -.1, .9], [-.8, .6, .2], [9., 9., 9.]])
    mask = search.export_mask()
    _assert_gpu(mask)
    retrain = cls.from_mask(initial, mask, padding_idx=2)
    expected_mask = np.array([[True, False, True], [True, True, False], [False, False, False]])
    np.testing.assert_array_equal(_array(retrain.export_mask()), expected_mask)
    np.testing.assert_allclose(_array(retrain.effective_weight()), [[.6, 0., .2], [.4, 0., 0.], [0., 0., 0.]])
    assert retrain.threshold_logits is None
    assert int(_array(retrain.nonzero_count())) == 3
    if gpu == "pytorch":
        import torch
        assert len(list(retrain.parameters())) == 1
        optimizer = torch.optim.SGD(retrain.parameters(), lr=.1, momentum=.9, weight_decay=.1)
    else:
        import tensorflow as tf
        assert len(retrain.trainable_variables) == 1
        optimizer = tf.keras.optimizers.SGD(.1, momentum=.9, weight_decay=.1)
    ids = _tensor(gpu, [0, 1, 2], "int64")
    for _ in range(2):
        gradient = _sum_gradient(gpu, retrain, ids)
        np.testing.assert_array_equal(_array(gradient), expected_mask)
        if gpu == "pytorch":
            optimizer.step()
        else:
            optimizer.apply_gradients([(gradient, retrain.weight)])
        effective = retrain.effective_weight()
        _assert_gpu(effective)
        assert _array(effective)[1, 1] < 0  # An active zero can learn.
        np.testing.assert_array_equal(_array(effective)[~expected_mask], 0)
        np.testing.assert_array_equal(_array(retrain.export_mask()), expected_mask)
    # Raw updates cannot reactivate excluded coordinates or the padding row.
    _assign(gpu, retrain.weight, np.ones((3, 3)))
    np.testing.assert_array_equal(_array(retrain(ids)), expected_mask)
    np.testing.assert_array_equal(_array(initial), np.array([[.6, -.8, .2], [.4, 0., .1], [9., 9., 9.]], np.float32))


def test_pruning_all_pruned_gpu_has_zero_output_and_zero_gradient(gpu):
    cls = _class(gpu, "embedding_pruning", "PrunableEmbedding")
    layer = cls.from_mask(_tensor(gpu, [[1., -2., 3.]]), _tensor(gpu, [[False] * 3], "bool"))
    gradient = _sum_gradient(gpu, layer, _tensor(gpu, [0, 0], "int64"))
    np.testing.assert_array_equal(_array(gradient), 0)
    np.testing.assert_array_equal(_array(layer.effective_weight()), 0)
    assert int(_array(layer.nonzero_count())) == 0


@pytest.mark.parametrize("mode", ["search", "retrain"])
def test_pruning_gpu_same_mode_checkpoint_restores_weights_support_and_padding(gpu, mode, tmp_path):
    cls = _class(gpu, "embedding_pruning", "PrunableEmbedding")
    initial = _tensor(gpu, [[.8, -.9], [2., 3.]])
    if mode == "search":
        source = cls(initial, threshold_init=0., padding_idx=1)
        target = cls(_tensor(gpu, np.zeros((2, 2))), threshold_init=3.)
        expected = [[.3, -.4], [0., 0.]]
    else:
        source = cls.from_mask(initial, _tensor(gpu, [[True, False], [True, True]], "bool"), padding_idx=1)
        target = cls.from_mask(_tensor(gpu, np.zeros((2, 2))), _tensor(gpu, [[True] * 2] * 2, "bool"))
        expected = [[.8, 0.], [0., 0.]]
    ids = _tensor(gpu, [0, 1], "int64")
    source(ids)
    target(ids)
    if gpu == "pytorch":
        import torch
        path = tmp_path / "embedding.pt"
        torch.save(source.state_dict(), path)
        target.load_state_dict(torch.load(path, map_location="cuda:0", weights_only=True))
    else:
        import tensorflow as tf
        path = tf.train.Checkpoint(layer=source).write(str(tmp_path / "embedding"))
        tf.train.Checkpoint(layer=target).read(path).assert_consumed()
    output = target(ids)
    for value in (target.weight, target.mask, output, target.export_mask()):
        _assert_gpu(value)
    np.testing.assert_allclose(_array(output), expected, atol=1e-6)
    np.testing.assert_array_equal(_array(target.export_mask()), np.asarray(expected) != 0)
    if mode == "search":
        _assert_gpu(target.threshold_logits)
        np.testing.assert_array_equal(_array(target.threshold_logits), [[0.]])


def _draw(framework, counts, eligible, positives, *, settings=None, seed=19, dtype="float32"):
    model = _class(framework, "negative_sampling", "ConditionalNegativeSampler")(**(settings or {}))
    result = model(_tensor(framework, counts, dtype), _tensor(framework, eligible, "bool"),
                   _tensor(framework, positives, "bool"), num_samples=512, seed=seed)
    for value in result.values():
        _assert_gpu(value)
    ids, probabilities = result["item_ids"], result["sampling_probs"]
    assert "int64" in str(ids.dtype)
    assert dtype in str(probabilities.dtype)
    assert tuple(ids.shape) == tuple(probabilities.shape) == (len(eligible), 512)
    return _array(ids), _array(probabilities)


@pytest.mark.parametrize("settings,expected", [
    ({}, [[0., .5, .5, 0.], [0., 0., .5, .5]]),
    ({"distribution": "popularity", "alpha": .5}, [[0., .2, .8, 0.], [0., 0., 2/3, 1/3]]),
    ({"distribution": "popularity", "alpha": .5, "uniform_mix": .5},
     [[0., .35, .65, 0.], [0., 0., 13/22, 9/22]]),
])
def test_sampler_gpu_returns_conditional_q_and_excludes_disallowed_items(gpu, settings, expected):
    ids, probabilities = _draw(gpu, [9., 1., 16., 4.],
        [[True, True, True, False], [False, True, True, True]],
        [[True, False, False, False], [False, True, False, False]], settings=settings)
    assert set(ids[0]) == {1, 2}
    assert set(ids[1]) == {2, 3}
    np.testing.assert_allclose(probabilities, np.take_along_axis(np.asarray(expected), ids, axis=1), rtol=1e-6)


def test_sampler_gpu_seed_reproduces_draws_and_preserves_global_random_stream(gpu):
    if gpu == "pytorch":
        import torch
        cpu_state = torch.random.get_rng_state().clone()
        gpu_state = torch.cuda.get_rng_state().clone()
    else:
        import tensorflow as tf
        tf.random.set_seed(731)
        expected_stream = tf.random.uniform([12])
        tf.random.set_seed(731)
    args = (gpu, [1.] * 5, [[True] * 5], [[False] * 5])
    first = _draw(*args, seed=2**62 + 17)
    second = _draw(*args, seed=2**62 + 17)
    third = _draw(*args, seed=2**62 + 18)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], third[0])
    if gpu == "pytorch":
        assert torch.equal(torch.random.get_rng_state(), cpu_state)
        assert torch.equal(torch.cuda.get_rng_state(), gpu_state)
    else:
        np.testing.assert_array_equal(tf.random.uniform([12]).numpy(), expected_stream.numpy())


@pytest.mark.parametrize("dtype,high,low", [("float32", 1e30, 1e-30), ("float64", 1e300, 1e-300)])
@pytest.mark.parametrize("mixture", [False, True], ids=["huge_alpha_tail", "catalog_mixture_tail"])
def test_sampler_gpu_keeps_extreme_tail_support_after_exclusion(gpu, dtype, high, low, mixture):
    settings = {"distribution": "popularity", "alpha": .5 if mixture else 1e308,
                "uniform_mix": 3 * low if mixture else 0.}
    counts = [high, low, 4 * low if mixture else low]
    ids, probabilities = _draw(gpu, counts, [[False, True, False], [False, True, True]],
        [[False] * 3] * 2, settings=settings, dtype=dtype)
    np.testing.assert_array_equal(ids[0], 1)
    np.testing.assert_array_equal(probabilities[0], 1.)
    assert set(ids[1]) == {1, 2}
    expected = [0., .4, .6] if mixture else [0., .5, .5]
    np.testing.assert_allclose(probabilities[1], np.take(expected, ids[1]), rtol=1e-6)


def test_sampler_gpu_tiny_catalog_uniform_mix_can_support_zero_count_item(gpu):
    ids, probabilities = _draw(gpu, [1., 0.], [[False, True]], [[False, False]],
        settings={"distribution": "popularity", "uniform_mix": 1e-100})
    np.testing.assert_array_equal(ids, 1)
    np.testing.assert_array_equal(probabilities, 1.)


@pytest.mark.parametrize("counts,eligible,positives,settings", [
    ([1., 2.], [[True, True]], [[True, True]], {}),
    ([0., 2.], [[True, False]], [[False, False]], {"distribution": "popularity"}),
    ([0., 0.], [[True, True]], [[False, False]], {"distribution": "popularity", "uniform_mix": .5}),
])
def test_sampler_gpu_rejects_undefined_support(gpu, counts, eligible, positives, settings):
    error = ValueError
    if gpu == "tensorflow":
        import tensorflow as tf
        error = (ValueError, tf.errors.InvalidArgumentError)
    with pytest.raises(error):
        _draw(gpu, counts, eligible, positives, settings=settings)
