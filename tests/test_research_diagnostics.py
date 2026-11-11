"""Analytic contracts for standalone representation and screening utilities."""

import importlib
from pathlib import Path

import numpy as np
import pytest


def _module(framework):
    return importlib.import_module(
        f"model_evo_harness.models.{framework}.research_diagnostics")


def _tensor(framework, values):
    if framework == "pytorch":
        import torch
        return torch.tensor(np.asarray(values))
    import tensorflow as tf
    return tf.constant(np.asarray(values))


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value.numpy()


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_rank_definitions_and_population_variance(framework):
    diagnose = _module(framework).RepresentationDiagnostics()
    x = _tensor(framework, np.array([
        [[1., 0.], [0., 1.]],
        [[1., 2.], [1., 2.]],
        [[0., 0.], [0., 0.]],
        [[3., 0.], [0., 1.]],
    ], np.float64))
    result = diagnose(x)
    expected_entropy_rank = np.exp(-.75 * np.log(.75) - .25 * np.log(.25))
    np.testing.assert_allclose(_array(result["effective_rank"]),
                               [2., 1., 0., expected_entropy_rank], atol=1e-12)
    np.testing.assert_allclose(_array(result["stable_rank"]), [2., 1., 0., 10 / 9], atol=1e-12)
    np.testing.assert_allclose(_array(result["output_variance"]), [.25, 0., 0., 1.25])
    np.testing.assert_array_equal(_array(result["valid_tokens"]), [2, 2, 2, 2])
    centered = _module(framework).RepresentationDiagnostics(center=True)(x)
    np.testing.assert_allclose(_array(centered["effective_rank"]), [1, 0, 0, 1], atol=1e-12)
    np.testing.assert_array_equal(_array(centered["output_variance"]),
                                  _array(result["output_variance"]))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_masks_sanitize_nonfinite_padding_and_zero_valid_samples(framework):
    diagnose = _module(framework).RepresentationDiagnostics()
    x = _tensor(framework, np.array([
        [[1., 0.], [np.nan, np.inf]],
        [[np.nan, np.inf], [np.inf, np.nan]],
    ], np.float32))
    mask = _tensor(framework, [[True, False], [False, False]])
    result = diagnose(x, mask, reference=x)
    for key, expected in {"effective_rank": [1, 0], "stable_rank": [1, 0],
                          "output_variance": [0, 0], "valid_tokens": [1, 0],
                          "branch_similarity": [1, 0]}.items():
        np.testing.assert_allclose(_array(result[key]), expected)
    centered = _module(framework).RepresentationDiagnostics(center=True)(x, mask, x)
    np.testing.assert_array_equal(_array(centered["effective_rank"]), [0, 0])
    np.testing.assert_array_equal(_array(centered["branch_similarity"]), [0, 0])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_centering_identical_fractional_tokens_does_not_create_spurious_rank(framework):
    x = _tensor(framework, np.tile(np.array([.1, .2, 2.3], np.float32), (1, 7, 1)))
    result = _module(framework).RepresentationDiagnostics(center=True)(x, reference=x)
    for key in ("effective_rank", "stable_rank", "output_variance", "branch_similarity"):
        np.testing.assert_array_equal(_array(result[key]), [0])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_branch_similarity_and_named_layer_inspection(framework):
    diagnose = _module(framework).RepresentationDiagnostics()
    x = _tensor(framework, np.array([[[1., 0.]], [[1., 0.]], [[1., 0.]]], np.float32))
    y = _tensor(framework, np.array([[[1., 0.]], [[0., 1.]], [[-1., 0.]]], np.float32))
    result = diagnose.inspect({"early": x, "late": y}, references={"early": y})
    assert set(result) == {"early", "late"}
    np.testing.assert_allclose(_array(result["early"]["branch_similarity"]), [1, 0, -1])
    assert "branch_similarity" not in result["late"]
    assert set(diagnose.inspect({"one": x}, valid_masks={"one": _tensor(
        framework, [[False], [True], [True]])})) == {"one"}


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_ranks_are_scale_invariant_and_empty_token_axis_is_defined(framework):
    diagnose = _module(framework).RepresentationDiagnostics()
    x = np.array([[[2., 0.], [0., 1.]]], np.float32)
    base = diagnose(_tensor(framework, x))
    for scale in (1e-20, 1e20):
        scaled = diagnose(_tensor(framework, x * scale))
        for key in ("effective_rank", "stable_rank"):
            np.testing.assert_allclose(_array(scaled[key]), _array(base[key]), rtol=1e-6)
    result = diagnose(_tensor(framework, np.zeros((2, 0, 3), np.float32)))
    for value in result.values():
        np.testing.assert_array_equal(_array(value), [0, 0])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_gradient_saliency_and_redundancy_are_explicit_evidence(framework):
    select = _module(framework).GradientFeatureSelector()
    # Field 0 and field 1 are duplicated; field 2 is orthogonal over B,D.
    x = np.array([[[1., 0.], [2., 0.], [0., 3.]],
                  [[2., 0.], [4., 0.], [0., 6.]]], np.float32)
    g = np.array([[[2., 0.], [1., 0.], [0., -1.]],
                  [[2., 0.], [1., 0.], [0., -1.]]], np.float32)
    result = select(_tensor(framework, x), _tensor(framework, g))
    np.testing.assert_allclose(_array(result["saliency"]), [3, 3, 4.5])
    np.testing.assert_allclose(_array(result["redundancy"]), [[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(_array(result["valid_counts"]), [2, 2, 2])
    np.testing.assert_array_equal(_array(result["overlap_counts"]), np.full((3, 3), 2))
    # Saliency is abs(sum over channels), not sum(abs(channel products)).
    cancel = select(_tensor(framework, [[[1., 1.]]]), _tensor(framework, [[[1., -1.]]]))
    np.testing.assert_array_equal(_array(cancel["saliency"]), [0])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_selection_masks_use_pairwise_overlap_and_ignore_nan_padding(framework):
    select = _module(framework).GradientFeatureSelector()
    x = np.array([[[1.], [1.], [np.nan]], [[2.], [np.inf], [np.nan]],
                  [[np.nan], [3.], [np.nan]]], np.float32)
    mask = np.array([[True, True, False], [True, False, False], [False, True, False]])
    result = select(_tensor(framework, x), _tensor(framework, np.ones_like(x)),
                    _tensor(framework, mask))
    np.testing.assert_array_equal(_array(result["saliency"]), [1.5, 2., 0])
    np.testing.assert_allclose(_array(result["redundancy"]), [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    np.testing.assert_array_equal(_array(result["overlap_counts"]), [[2, 1, 0], [1, 2, 0], [0, 0, 0]])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_budget_ties_and_interaction_only_fields_are_protected_atomically(framework):
    select = _module(framework).GradientFeatureSelector()
    scores = _tensor(framework, [9., 9., 1., 0., 0.])
    assert select.select(scores, 2) == (0, 1)
    assert select.select(scores, 0) == ()
    assert select.select(scores, 3, protected_groups=[(3, 4)]) == (0, 3, 4)
    assert select.select(scores, 4, protected_indices=[2], protected_groups=[(3, 4)]) == (0, 2, 3, 4)
    assert select.select(scores, 3, protected_groups=[(2, 3), (3, 4)]) == (2, 3, 4)
    with pytest.raises(ValueError, match="protected"):
        select.select(scores, 1, protected_groups=[(3, 4)])
    for budget in (-1, 6, 1.5):
        with pytest.raises((ValueError, TypeError)):
            select.select(scores, budget)
    with pytest.raises((ValueError, IndexError)):
        select.select(scores, 2, protected_indices=[5])
    with pytest.raises(ValueError, match="finite"):
        select.select(_tensor(framework, [1., np.nan]), 1)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_invalid_shapes_and_valid_nonfinite_values_are_rejected(framework):
    module = _module(framework)
    x = _tensor(framework, np.ones((2, 3, 2), np.float32))
    errors = (ValueError, TypeError)
    if framework == "tensorflow":
        import tensorflow as tf
        errors += (tf.errors.InvalidArgumentError,)
    with pytest.raises(errors):
        module.RepresentationDiagnostics()(_tensor(framework, np.ones((2, 3), np.float32)))
    with pytest.raises(errors):
        module.RepresentationDiagnostics()(x, _tensor(framework, np.ones((2, 1), bool)))
    with pytest.raises(errors):
        module.RepresentationDiagnostics()(_tensor(framework, [[[np.nan]]]))
    with pytest.raises(errors):
        module.RepresentationDiagnostics()(x, reference=_tensor(framework, np.ones((2, 3, 1), np.float32)))
    with pytest.raises(errors):
        module.GradientFeatureSelector()(x, _tensor(framework, np.ones((2, 3, 1), np.float32)))
    with pytest.raises(errors):
        module.GradientFeatureSelector()(x, x, _tensor(framework, np.ones((2, 3), np.float32)))


def test_pytorch_metrics_detach_from_actual_objective_gradients():
    import torch
    module = _module("pytorch")
    x = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
    gradient, = torch.autograd.grad((x.square()).sum(), x, create_graph=True)
    diagnostics = module.RepresentationDiagnostics()(x, reference=x)
    evidence = module.GradientFeatureSelector()(x, gradient)
    np.testing.assert_allclose(_array(evidence["saliency"]), [10, 50])
    assert all(not value.requires_grad for value in {**diagnostics, **evidence}.values())


def test_tensorflow_metrics_detach_from_actual_objective_gradients():
    import tensorflow as tf
    module = _module("tensorflow")
    x = tf.Variable([[[1., 2.], [3., 4.]]])
    with tf.GradientTape(persistent=True) as outer:
        with tf.GradientTape() as inner:
            loss = tf.reduce_sum(tf.square(x))
        gradient = inner.gradient(loss, x)
        diagnostics = module.RepresentationDiagnostics()(x, reference=x)
        evidence = module.GradientFeatureSelector()(x, gradient)
        total = tf.add_n([tf.reduce_sum(value) for value in (
            diagnostics["effective_rank"], diagnostics["output_variance"], evidence["saliency"])])
    np.testing.assert_allclose(_array(evidence["saliency"]), [10, 50])
    assert outer.gradient(total, x) is None


def test_framework_agreement_for_centered_masked_inputs():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(4, 5, 3)).astype(np.float32)
    g = rng.normal(size=x.shape).astype(np.float32)
    mask = rng.uniform(size=(4, 5)) > .3
    x[~mask] = np.nan
    g[~mask] = np.nan
    outputs = []
    for framework in ("pytorch", "tensorflow"):
        module = _module(framework)
        tx, tg, tm = (_tensor(framework, value) for value in (x, g, mask))
        outputs.append({**module.RepresentationDiagnostics(center=True)(tx, tm, tg),
                        **module.GradientFeatureSelector()(tx, tg, tm)})
    for key in outputs[0]:
        np.testing.assert_allclose(_array(outputs[0][key]), _array(outputs[1][key]), atol=2e-6)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_standalone_source_exec_without_package_imports(framework):
    path = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models" / framework / "research_diagnostics.py"
    namespace = {"__name__": "standalone_research_diagnostics"}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    x = _tensor(framework, np.array([[[1., 0.], [0., 1.]]], np.float32))
    np.testing.assert_allclose(_array(namespace["RepresentationDiagnostics"]()(x)["effective_rank"]), [2])
    result = namespace["GradientFeatureSelector"]()(x, x)
    assert namespace["GradientFeatureSelector"]().select(result["saliency"], 1) == (0,)


def test_tensorflow_tensor_metrics_work_in_graph_mode():
    import tensorflow as tf
    module = _module("tensorflow")
    diagnose, select = module.RepresentationDiagnostics(center=True), module.GradientFeatureSelector()

    @tf.function(input_signature=[tf.TensorSpec((None, None, 2), tf.float32),
                                 tf.TensorSpec((None, None), tf.bool)])
    def metrics(x, mask):
        return diagnose(x, mask), select(x, x, mask)

    diagnostics, evidence = metrics(tf.constant([[[1., 0.], [0., 1.]]]), tf.constant([[True, True]]))
    np.testing.assert_allclose(_array(diagnostics["effective_rank"]), [1], atol=1e-6)
    np.testing.assert_allclose(_array(evidence["saliency"]), [1, 1])


def test_screening_avoids_integer_gemm_for_cuda_portability():
    """CUDA GEMM kernels do not support Long; exercise the real tensor path."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class FloatingGemmOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
                assert args[0].is_floating_point(), "integer GEMM has no CUDA kernel"
            return func(*args, **(kwargs or {}))

    values = torch.ones((3, 2, 1))
    mask = torch.tensor([[True, False], [True, True], [False, True]])
    with FloatingGemmOnly():
        result = _module("pytorch").GradientFeatureSelector()(values, values, mask)
    assert result["overlap_counts"].dtype == torch.int64
    np.testing.assert_array_equal(_array(result["overlap_counts"]), [[2, 1], [1, 2]])


def test_tensorflow_count_graph_has_no_cpu_only_int64_matmul():
    import tensorflow as tf
    selector = _module("tensorflow").GradientFeatureSelector()

    @tf.function(autograph=False)
    def compute(values, mask):
        return selector(values, values, mask)["overlap_counts"]

    graph = compute.get_concrete_function(tf.TensorSpec((None, 2, 1), tf.float32),
                                         tf.TensorSpec((None, 2), tf.bool))
    for operation in graph.graph.get_operations():
        if operation.type in {"MatMul", "BatchMatMulV2"}:
            assert operation.inputs[0].dtype.is_floating, "int64 MatMul is CPU-only"
    result = compute(tf.ones((3, 2, 1)), tf.constant([[True, False], [True, True], [False, True]]))
    assert result.dtype == tf.int64
    np.testing.assert_array_equal(result.numpy(), [[2, 1], [1, 2]])
