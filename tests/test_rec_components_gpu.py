"""Device assertions for released mechanisms; no dataset training is performed."""
import importlib

import pytest


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
@pytest.mark.parametrize("mechanism", ["coaction", "comirec_sa", "comirec_dr", "cin", "sdim", "deepcrossing"])
def test_rec_components_forward_and_backward_execute_on_gpu(framework, mechanism):
    if framework == "pytorch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU unavailable")
        device = torch.device("cuda:0")
        history = torch.tensor([[[1., 2.], [3., 4.], [float("nan"), float("nan")]]], device=device, requires_grad=True)
        valid = torch.tensor([[True, True, False]], device=device)
        args, tracked = (history, valid), [history]
        if mechanism == "coaction":
            module = importlib.import_module("model_evo_harness.models.pytorch.coaction")
            model = module.CoAction(2, [3, 2], order=2)
            theta = torch.full((1, model.parameter_dim), .2, device=device, requires_grad=True)
            args, tracked = (theta, history, valid), [theta, history]
        elif mechanism.startswith("comirec"):
            module = importlib.import_module("model_evo_harness.models.pytorch.multi_interest")
            model = module.ComiRecSA(2, 2, max_length=3) if mechanism.endswith("sa") else module.ComiRecDR(2, 2, 3)
        elif mechanism == "cin":
            module = importlib.import_module("model_evo_harness.models.pytorch.interactions")
            model = module.XDeepFM([3, 3], 2, cin_activation="identity")
            args, tracked = (torch.tensor([[0, 1]], device=device),), []
        elif mechanism == "sdim":
            module = importlib.import_module("model_evo_harness.models.pytorch.history_retrieval")
            model = module.SDIMCollisionPooling(2, normalize_per_hash=True)
            args = (torch.tensor([[1., 2.]], device=device), history, valid)
        else:
            module = importlib.import_module("model_evo_harness.models.pytorch.paper_ranking")
            model = module.DeepCrossing(2, 3)
            args = (history[:, 0],)
        model = model.to(device)
        result = model(*args)
        output = result.get("interests", result.get("representation")) if isinstance(result, dict) else result
        assert output.is_cuda and torch.isfinite(output).all()
        output.square().sum().backward()
        gradients = [x.grad for x in tracked + list(model.parameters())]
        assert gradients and all(g is not None and g.is_cuda and torch.isfinite(g).all() for g in gradients)
        if any(value is history for value in tracked):
            assert torch.equal(history.grad[:, 2], torch.zeros_like(history.grad[:, 2]))
    else:
        tf = pytest.importorskip("tensorflow")
        if not tf.config.list_physical_devices("GPU"):
            pytest.skip("TensorFlow GPU unavailable")
        with tf.device("/GPU:0"):
            history = tf.Variable([[[1., 2.], [3., 4.], [float("nan"), float("nan")]]])
            valid = tf.constant([[True, True, False]])
            args, tracked = (history, valid), [history]
            if mechanism == "coaction":
                module = importlib.import_module("model_evo_harness.models.tensorflow.coaction")
                model = module.CoAction(2, [3, 2], order=2)
                theta = tf.Variable(tf.fill((1, model.parameter_dim), .2))
                args, tracked = (theta, history, valid), [theta, history]
            elif mechanism.startswith("comirec"):
                module = importlib.import_module("model_evo_harness.models.tensorflow.multi_interest")
                model = module.ComiRecSA(2, 2, max_length=3) if mechanism.endswith("sa") else module.ComiRecDR(2, 2, 3)
            elif mechanism == "cin":
                module = importlib.import_module("model_evo_harness.models.tensorflow.interactions")
                model = module.XDeepFM([3, 3], 2, cin_activation="identity")
                args, tracked = (tf.constant([[0, 1]]),), []
            elif mechanism == "sdim":
                module = importlib.import_module("model_evo_harness.models.tensorflow.history_retrieval")
                model = module.SDIMCollisionPooling(2, normalize_per_hash=True)
                args = (tf.constant([[1., 2.]]), history, valid)
            else:
                module = importlib.import_module("model_evo_harness.models.tensorflow.paper_ranking")
                model = module.DeepCrossing(2, 3)
                args = (history[:, 0],)
            with tf.GradientTape() as tape:
                # Slice inside the tape so DeepCrossing input lineage is observable.
                if mechanism == "deepcrossing":
                    args = (history[:, 0],)
                result = model(*args)
                output = result.get("interests", result.get("representation")) if isinstance(result, dict) else result
                loss = tf.reduce_sum(tf.square(output))
            assert "GPU:0" in output.device and bool(tf.reduce_all(tf.math.is_finite(output)))
            gradients = tape.gradient(loss, tracked + model.trainable_variables)
            assert gradients and all(g is not None for g in gradients)
            for gradient in gradients:
                values = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
                assert "GPU:0" in values.device and bool(tf.reduce_all(tf.math.is_finite(values)))
            if any(value is history for value in tracked):
                assert bool(tf.reduce_all(gradients[next(i for i, value in enumerate(tracked) if value is history)][:, 2] == 0))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_sdim_half_precision_normalization_stays_finite_on_gpu(framework):
    if framework == "pytorch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU unavailable")
        from model_evo_harness.models.pytorch.history_retrieval import SDIMCollisionPooling
        model = SDIMCollisionPooling(2, 1, 2, normalize_per_hash=True).cuda().half()
        with torch.no_grad():
            model.rotations.copy_(torch.eye(2, device="cuda")[None])
        for magnitude in (0., 1e-7):
            history = torch.tensor([[[magnitude, 0.], [float("nan"), float("nan")]]], device="cuda", dtype=torch.float16, requires_grad=True)
            result = model(torch.ones((1, 2), device="cuda", dtype=torch.float16), history,
                           torch.tensor([[True, False]], device="cuda"))["representation"]
            result.sum().backward()
            assert result.is_cuda and result.dtype == torch.float16
            assert torch.isfinite(result).all() and torch.isfinite(history.grad).all()
            assert torch.count_nonzero(history.grad[:, 1]) == 0
    else:
        tf = pytest.importorskip("tensorflow")
        if not tf.config.list_physical_devices("GPU"):
            pytest.skip("TensorFlow GPU unavailable")
        from model_evo_harness.models.tensorflow.history_retrieval import SDIMCollisionPooling
        policy = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            with tf.device("/GPU:0"):
                model = SDIMCollisionPooling(2, 1, 2, normalize_per_hash=True)
                model.rotations.assign(tf.eye(2)[None])
                for magnitude in (0., 1e-7):
                    history = tf.Variable([[[magnitude, 0.], [float("nan"), float("nan")]]], dtype=tf.float16)
                    with tf.GradientTape() as tape:
                        result = model(tf.ones((1, 2), tf.float16), history,
                                       tf.constant([[True, False]]))["representation"]
                        loss = tf.reduce_sum(result)
                    gradient = tape.gradient(loss, history)
                    assert "GPU:0" in result.device and result.dtype == tf.float16
                    assert "GPU:0" in gradient.device and bool(tf.reduce_all(tf.math.is_finite(gradient)))
                    assert bool(tf.reduce_all(tf.math.is_finite(result)))
                    assert bool(tf.reduce_all(gradient[:, 1] == 0))
        finally:
            tf.keras.mixed_precision.set_global_policy(policy)
