"""GPU execution of the discount branch and its purchase-only training objective."""

import numpy as np
import pytest


def _inputs(length, purchases=3):
    history = np.full((2, length), .4, np.float32)
    history_mask = np.ones((2, length), bool)
    history_mask[0, -1] = False
    history_mask[1] = False
    history[~history_mask] = np.nan
    purchase_mask = np.ones((2, purchases), bool)
    if purchases:
        purchase_mask[0, -1] = False
    purchase_mask[1] = False
    items = np.full((2, purchases, 2), .3, np.float32)
    attributes = np.full((2, purchases, 3), .2, np.float32)
    items[~purchase_mask] = np.nan
    attributes[~purchase_mask] = np.nan
    return [history, history_mask, np.full((2, 2), .2, np.float32), items,
            attributes, purchase_mask, np.full((2, 2), .1, np.float32)]


@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("length,purchases", [(3, 3), (6, 3), (5, 0)])
def test_discount_training_pytorch_gpu(precision, length, purchases, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    from model_evo_harness.models.pytorch.discount_aware import (
        DiscountPreferenceBranch, DiscountAwareConversionHead,
    )

    fft_calls = []
    for name in ("rfft", "irfft"):
        operation = getattr(torch.fft, name)
        def record_fft(*args, _operation=operation, **kwargs):
            result = _operation(*args, **kwargs)
            assert result.is_cuda
            fft_calls.append(result.dtype)
            return result
        monkeypatch.setattr(torch.fft, name, record_fft)

    arrays = _inputs(length, purchases)
    args = [torch.tensor(value, device="cuda", requires_grad=i not in (1, 5))
            for i, value in enumerate(arrays)]
    interest = torch.full((2, 2), .3, device="cuda", requires_grad=True)
    branch = DiscountPreferenceBranch(length, 2, 3, 2, 4).cuda()
    head = DiscountAwareConversionHead(2, 4 + 2 * length, 4).cuda()
    parameters = list(branch.parameters()) + list(head.parameters())
    with torch.no_grad():
        for parameter in parameters:
            parameter.copy_(torch.arange(parameter.numel(), device="cuda").reshape(parameter.shape) * .002 + .03)
    conversion = torch.tensor([1., 0.], device="cuda", requires_grad=True)
    discounts = torch.tensor([.25, float("nan")], device="cuda", requires_grad=True)
    observed = torch.tensor([True, True], device="cuda")
    with torch.autocast("cuda", dtype=getattr(torch, precision), enabled=precision != "float32"):
        features = branch(*args)
        outputs = head(interest, features["representation"])
        losses = head.objective(outputs, conversion, discounts, observed, alpha=.4)
    assert outputs["conversion_logit"].dtype == getattr(torch, precision)
    assert fft_calls[:2] == [torch.complex64, torch.float32]
    assert all(losses[name].dtype == torch.float32 for name in ("loss", "conversion_loss", "discount_loss"))
    for value in [*features.values(), *outputs.values(), *losses.values()]:
        assert value.is_cuda and torch.isfinite(value).all()
    torch.testing.assert_close(features["representation"][1], torch.zeros(4 + 2 * length, device="cuda"))
    torch.testing.assert_close(features["filtered_history"] + features["residual_history"],
                               torch.nan_to_num(args[0]), atol=2e-6, rtol=2e-5)
    assert features["attention"].shape == (2, purchases)
    assert losses["discount_count"].item() == 1
    torch.testing.assert_close(losses["discount_loss"], (outputs["discount_prediction"][0].float() - .25).square())
    losses["loss"].backward()
    tracked = [args[i] for i in (0, 2, 3, 4, 6)] + [interest] + parameters
    assert all(value.grad is not None and value.grad.is_cuda and torch.isfinite(value.grad).all()
               for value in tracked)
    for index, mask_index in ((0, 1), (3, 5), (4, 5)):
        assert torch.count_nonzero(args[index].grad[~args[mask_index]]) == 0
    assert torch.count_nonzero(args[0].grad[0]) > 0
    assert torch.count_nonzero(interest.grad) > 0
    assert conversion.grad is None and discounts.grad is None
    # An actual optimizer update exercises parameter gradients, beyond a forward-only probe.
    before = head.conversion_head[-1].weight.detach().clone()
    torch.optim.SGD(parameters, lr=.01).step()
    assert not torch.equal(before, head.conversion_head[-1].weight)


@pytest.mark.parametrize("precision", ["float32", "mixed_float16", "mixed_bfloat16"])
@pytest.mark.parametrize("length,purchases", [(3, 3), (6, 3), (5, 0)])
def test_discount_training_tensorflow_gpu(precision, length, purchases, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("TensorFlow GPU unavailable")
    from model_evo_harness.models.tensorflow.discount_aware import (
        DiscountPreferenceBranch, DiscountAwareConversionHead,
    )

    fft_calls = []
    for name in ("rfft", "irfft"):
        operation = getattr(tf.signal, name)
        def record_fft(*args, _operation=operation, **kwargs):
            result = _operation(*args, **kwargs)
            if tf.executing_eagerly():
                assert "GPU:0" in result.device
                fft_calls.append(result.dtype)
            return result
        monkeypatch.setattr(tf.signal, name, record_fft)

    policy = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy(precision)
        with tf.device("/GPU:0"):
            args = [tf.constant(value) if i in (1, 5) else tf.Variable(value)
                    for i, value in enumerate(_inputs(length, purchases))]
            interest = tf.Variable(tf.fill((2, 2), .3))
            branch = DiscountPreferenceBranch(length, 2, 3, 2, 4)
            head = DiscountAwareConversionHead(2, 4 + 2 * length, 4)
            head(interest, branch(*args)["representation"])
            parameters = branch.trainable_variables + head.trainable_variables
            for parameter in parameters:
                parameter.assign(tf.reshape(tf.range(tf.size(parameter), dtype=tf.float32), parameter.shape) * .002 + .03)
            conversion = tf.Variable([1., 0.])
            discounts = tf.Variable([.25, np.nan])
            with tf.GradientTape() as tape:
                features = branch(*args)
                outputs = head(interest, features["representation"])
                losses = head.objective(outputs, conversion, discounts, tf.constant([True, True]), alpha=.4)
            assert outputs["conversion_logit"].dtype == tf.as_dtype(tf.keras.mixed_precision.global_policy().compute_dtype)
            assert fft_calls[:2] == [tf.complex64, tf.float32]
            assert all(losses[name].dtype == tf.float32 for name in ("loss", "conversion_loss", "discount_loss"))
            for value in [*features.values(), *outputs.values(), *losses.values()]:
                if value.dtype.is_floating:
                    assert "GPU:0" in value.device
                    assert bool(tf.reduce_all(tf.math.is_finite(value)))
            np.testing.assert_array_equal(features["representation"][1].numpy(), 0)
            np.testing.assert_allclose(tf.cast(features["filtered_history"] + features["residual_history"], tf.float32).numpy(),
                                       np.nan_to_num(_inputs(length, purchases)[0]), atol=.005, rtol=.005)
            assert features["attention"].shape == (2, purchases)
            assert int(losses["discount_count"]) == 1
            np.testing.assert_allclose(losses["discount_loss"].numpy(),
                                       (float(outputs["discount_prediction"][0]) - .25) ** 2, atol=1e-6)
            tracked = [args[i] for i in (0, 2, 3, 4, 6)] + [interest] + parameters
            gradients = tape.gradient(losses["loss"], tracked + [conversion, discounts])
            assert gradients[-2:] == [None, None]
            gradients = gradients[:-2]
            assert all(g is not None and "GPU:0" in g.device and bool(tf.reduce_all(tf.math.is_finite(g)))
                       for g in gradients)
            for grad_index, mask_index in ((0, 1), (2, 5), (3, 5)):
                np.testing.assert_array_equal(gradients[grad_index].numpy()[~args[mask_index].numpy()], 0)
            assert np.count_nonzero(gradients[0][0].numpy()) > 0
            assert np.count_nonzero(gradients[5].numpy()) > 0
            before = head.conversion_head.layers[-1].kernel.numpy().copy()
            tf.keras.optimizers.SGD(.01).apply_gradients(zip(gradients[6:], parameters))
            assert not np.array_equal(before, head.conversion_head.layers[-1].kernel.numpy())
    finally:
        tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_discount_empty_auxiliary_gpu(framework):
    if framework == "pytorch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU unavailable")
        from model_evo_harness.models.pytorch.discount_aware import DiscountAwareConversionHead
        predictions = torch.tensor([.2, .8], device="cuda", requires_grad=True)
        outputs = {"conversion_logit": torch.zeros(2, device="cuda", requires_grad=True),
                   "discount_prediction": predictions}
        result = DiscountAwareConversionHead.objective(outputs, torch.zeros(2, device="cuda"),
                    torch.full((2,), float("nan"), device="cuda"), torch.ones(2, dtype=torch.bool, device="cuda"))
        result["discount_loss"].backward()
        assert result["discount_loss"].is_cuda and result["discount_loss"].item() == 0
        assert result["discount_count"].item() == 0
        assert predictions.grad.is_cuda and torch.count_nonzero(predictions.grad) == 0
    else:
        tf = pytest.importorskip("tensorflow")
        if not tf.config.list_physical_devices("GPU"):
            pytest.skip("TensorFlow GPU unavailable")
        from model_evo_harness.models.tensorflow.discount_aware import DiscountAwareConversionHead
        with tf.device("/GPU:0"):
            predictions = tf.Variable([.2, .8])
            head = DiscountAwareConversionHead(1, 1, 2)
            with tf.GradientTape() as tape:
                result = head.objective({"conversion_logit": tf.zeros(2), "discount_prediction": predictions},
                            tf.zeros(2), tf.fill((2,), np.nan), tf.ones(2, tf.bool))
            gradient = tape.gradient(result["discount_loss"], predictions)
            assert "GPU:0" in result["discount_loss"].device and float(result["discount_loss"]) == 0
            assert int(result["discount_count"]) == 0
            assert "GPU:0" in gradient.device and bool(tf.reduce_all(gradient == 0))


def test_discount_dynamic_graph_tensorflow_gpu():
    tf = pytest.importorskip("tensorflow")
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("TensorFlow GPU unavailable")
    from model_evo_harness.models.tensorflow.discount_aware import (
        DiscountPreferenceBranch, DiscountAwareConversionHead,
    )
    policy = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        with tf.device("/GPU:0"):
            branch = DiscountPreferenceBranch(5, 2, 3, 2, 4)
            head = DiscountAwareConversionHead(2, 14, 4)
            warm = [tf.constant(x) for x in _inputs(5)]
            head(tf.ones((2, 2)), branch(*warm)["representation"])

            @tf.function(input_signature=[
                tf.TensorSpec((None, 5), tf.float32), tf.TensorSpec((None, 5), tf.bool),
                tf.TensorSpec((None, None, 2), tf.float32),
                tf.TensorSpec((None, None, 3), tf.float32),
                tf.TensorSpec((None, None), tf.bool),
            ])
            def run(history, history_mask, items, attributes, purchase_mask):
                with tf.GradientTape() as tape:
                    tape.watch([history, items, attributes])
                    batch = tf.shape(history)[0]
                    fixed = tf.ones((batch, 2))
                    representation = branch(history, history_mask, fixed, items, attributes,
                                            purchase_mask, fixed)["representation"]
                    outputs = head(fixed, representation)
                    losses = head.objective(outputs, tf.zeros((batch,)), tf.fill((batch,), np.nan),
                                            tf.ones((batch,), tf.bool))
                gradients = tape.gradient(losses["loss"], [history, items, attributes] +
                                          branch.trainable_variables + head.trainable_variables)
                return representation, losses["loss"], gradients

            for batch, purchases in ((2, 3), (1, 0)):
                arrays = _inputs(5, purchases)
                args = [tf.constant(arrays[i][:batch]) for i in (0, 1, 3, 4, 5)]
                representation, loss, gradients = run(*args)
                assert representation.shape == (batch, 14)
                for value in [representation, loss, *gradients]:
                    assert value is not None and "GPU:0" in value.device
                    assert bool(tf.reduce_all(tf.math.is_finite(value)))
                for gradient, mask in zip(gradients[:3], [args[1], args[4], args[4]]):
                    np.testing.assert_array_equal(gradient.numpy()[~mask.numpy()], 0)
            assert run.experimental_get_tracing_count() == 1
    finally:
        tf.keras.mixed_precision.set_global_policy(policy)
