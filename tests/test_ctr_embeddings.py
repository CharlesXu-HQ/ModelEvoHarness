import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
@pytest.mark.parametrize("mode", ["init", "sum", "concat"])
def test_pretrained_fusion_padding_oov_and_freezing(framework, mode):
    m = importlib.import_module(f"model_evo_harness.models.{framework}.ctr_embeddings")
    weights = [[7., 7.], [1., 2.], [3., 4.], [5., 6.]]
    if framework == "pytorch":
        import torch
        ids = torch.tensor([0, 1, 3])
        layer = m.PretrainedIDFusion(weights, 2, mode=mode, freeze_pretrained=True)
        if mode != "init":
            with torch.no_grad():
                layer.id_embedding.weight.fill_(1.)
                if mode == "concat":
                    layer.projection.weight.copy_(torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]]))
        out = layer(ids).detach().numpy()
        assert not layer.pretrained.weight.requires_grad
    else:
        import tensorflow as tf
        ids = tf.constant([0, 1, 3])
        layer = m.PretrainedIDFusion(weights, 2, mode=mode, freeze_pretrained=True)
        layer(ids)
        if mode != "init":
            layer.id_embedding.embeddings.assign(tf.ones([4, 2]))
            if mode == "concat":
                layer.projection.kernel.assign([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        out = layer(ids).numpy()
        assert not layer.pretrained.trainable
    expected = np.array([[0., 0.], [1., 2.], [5., 6.]])
    if mode != "init":
        expected[1:] += 1
    np.testing.assert_allclose(out, expected)
    # Changing trainability alone never removes appended/pretrained-only IDs.
    tuned = m.PretrainedIDFusion(weights, 2, mode="init", freeze_pretrained=False)
    actual = tuned(ids)
    actual = actual.detach().numpy() if framework == "pytorch" else actual.numpy()
    np.testing.assert_allclose(actual, [[0., 0.], [1., 2.], [5., 6.]])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_dense_baseline_covers_linear_and_hidden_recipes(framework):
    m = importlib.import_module(f"model_evo_harness.models.{framework}.ctr_embeddings")
    if framework == "pytorch":
        import torch
        x = torch.ones(3, 2)
    else:
        import tensorflow as tf
        x = tf.ones([3, 2])
    for widths in ((), (4, 3)):
        layer = m.ResponseMLP(2, widths)
        assert tuple(layer(x).shape) == (3,)


@pytest.mark.parametrize('framework', ['pytorch', 'tensorflow'])
@pytest.mark.parametrize('mode', ['init', 'sum', 'concat'])
def test_pretrained_fusion_preserves_float64_for_identity_and_projection(framework, mode):
    m = importlib.import_module(f'model_evo_harness.models.{framework}.ctr_embeddings')
    values = np.array([[0., 0.], [1.123456789123, 2.]], np.float64)
    layer = m.PretrainedIDFusion(values, 2, mode=mode)
    if framework == 'pytorch':
        import torch
        actual = layer(torch.tensor([1]))
        assert actual.dtype == torch.float64
        if mode == 'init':
            np.testing.assert_array_equal(actual.detach().numpy(), values[1:])
    else:
        import tensorflow as tf
        actual = layer(tf.constant([1]))
        assert actual.dtype == tf.float64
        if mode == 'init':
            np.testing.assert_array_equal(actual.numpy(), values[1:])


@pytest.mark.parametrize('mode', ['init', 'sum', 'concat'])
@pytest.mark.parametrize('freeze', [True, False])
def test_mapped_fusion_values_padding_gradients_and_pretrained_only_rows(mode, freeze):
    import torch
    import tensorflow as tf
    pt = importlib.import_module('model_evo_harness.models.pytorch.ctr_embeddings')
    kt = importlib.import_module('model_evo_harness.models.tensorflow.ctr_embeddings')
    weights = np.array([[7., 7.], [1., 2.], [3., 4.], [5., 6.]], np.float64)
    source = pt.PretrainedIDFusion(weights, 3, mode, freeze)
    target = kt.PretrainedIDFusion(weights, 3, mode, freeze)
    target.projection.kernel.assign(source.projection.weight.detach().numpy().T)
    if mode != 'init':
        target.id_embedding.embeddings.assign(source.id_embedding.weight.detach().numpy())
    ids = np.array([[0, 1, 3], [3, 0, 2]], np.int64)
    ty = source(torch.tensor(ids))
    ty.square().sum().backward()
    variables = [target.pretrained.embeddings, target.projection.kernel]
    if mode != 'init': variables.append(target.id_embedding.embeddings)
    with tf.GradientTape() as tape:
        ky = target(tf.constant(ids))
        loss = tf.reduce_sum(ky ** 2)
    gradients = tape.gradient(loss, variables)
    np.testing.assert_allclose(ty.detach().numpy(), ky.numpy(), atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(ky.numpy()[ids == 0], 0)
    np.testing.assert_allclose(source.projection.weight.grad.numpy().T, gradients[1].numpy(), atol=1e-11, rtol=1e-11)
    if freeze:
        assert source.pretrained.weight.grad is None
        # Keras excludes frozen weights from optimizer variables, but a direct
        # GradientTape request may still differentiate its underlying resource.
        assert all(v is not target.pretrained.embeddings for v in target.trainable_variables)
    else:
        dense = tf.convert_to_tensor(gradients[0]).numpy()
        np.testing.assert_allclose(source.pretrained.weight.grad.numpy(), dense, atol=1e-11, rtol=1e-11)
        np.testing.assert_array_equal(dense[0], 0)
        assert np.count_nonzero(dense[3])  # Host-designated pretrained-only/OOV row remains trainable.
    if mode != 'init':
        dense = tf.convert_to_tensor(gradients[2]).numpy()
        np.testing.assert_allclose(source.id_embedding.weight.grad.numpy(), dense, atol=1e-11, rtol=1e-11)
        np.testing.assert_array_equal(dense[0], 0)
    before = target.pretrained.embeddings.numpy().copy()
    trainable_pairs = [(g, v) for g, v in zip(gradients, variables)
                       if any(v is trainable for trainable in target.trainable_variables)]
    tf.keras.optimizers.SGD(.01).apply_gradients(trainable_pairs)
    after = target.pretrained.embeddings.numpy()
    np.testing.assert_array_equal(after[0], before[0])
    if freeze:
        np.testing.assert_array_equal(after, before)
    else:
        assert not np.array_equal(after[3], before[3])


@pytest.mark.parametrize('widths', [(), (3, 4)])
def test_mapped_dense_baseline_values_and_input_gradients(widths):
    import torch
    import tensorflow as tf
    pt = importlib.import_module('model_evo_harness.models.pytorch.ctr_embeddings')
    kt = importlib.import_module('model_evo_harness.models.tensorflow.ctr_embeddings')
    source, target = pt.ResponseMLP(2, widths), kt.ResponseMLP(2, widths)
    for left, right in zip([layer for layer in source.network if isinstance(layer, torch.nn.Linear)], target.network.layers):
        right.kernel.assign(left.weight.detach().numpy().T)
        right.bias.assign(left.bias.detach().numpy())
    values = [[1., -2.], [2., 3.]]
    tx, kx = torch.tensor(values, requires_grad=True), tf.Variable(values)
    ty = source(tx)
    ty.square().sum().backward()
    with tf.GradientTape() as tape:
        ky = target(kx)
        loss = tf.reduce_sum(ky ** 2)
    np.testing.assert_allclose(ty.detach().numpy(), ky.numpy(), atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(tx.grad.numpy(), tape.gradient(loss, kx).numpy(), atol=1e-7, rtol=1e-6)
