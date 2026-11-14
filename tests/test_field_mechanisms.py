"""Formula and cross-framework contracts for remaining FuxiCTR field mechanisms."""

import importlib
from pathlib import Path

import numpy as np
import pytest


SPECS = [
    ("field_products", "FieldAwareFactorization", (3, 2)),
    ("field_products", "OperationAwareInteraction", (3, 2)),
    ("field_products", "HigherOrderFactorization", (3, 2, 3)),
    ("field_products", "InteractionMachine", (3, 2, 3)),
    ("field_products", "HolographicInteraction", (3, 2)),
    ("field_products", "DenseSparseInteraction", (3, 2)),
    ("field_products", "LorentzInteraction", (3, 2)),
    ("field_algebra", "LogarithmicInteraction", (3, 2, 2)),
    ("field_algebra", "EulerInteraction", (3, 2, 2)),
    ("field_algebra", "GeneralizedInteraction", (3, 2, 2, 2)),
    ("field_attention", "HierarchicalInteractionAggregation", (3, 2, 2)),
    ("field_attention", "FieldGraphInteraction", (3, 2)),
    ("field_attention", "DisentangledFieldAttention", (3, 2, 2)),
    ("field_attention", "SelfAttentionInteraction", (3, 2)),
    ("field_convolution", "OrderedKMaxConvolution", (3, 2, 2, 3, 2)),
    ("field_convolution", "ConvolutionalFieldGenerator", (3, 2, 2, 3, 2, 2)),
]


def cls(framework, module, name):
    pytest.importorskip("torch" if framework == "pytorch" else "tensorflow")
    path = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models" / framework / f"{module}.py"
    assert path.is_file(), f"Missing standalone field mechanism: {path}"
    return getattr(importlib.import_module(f"model_evo_harness.models.{framework}.{module}"), name)


def tensor(framework, value):
    value = np.asarray(value)
    if value.dtype != bool:
        value = value.astype("float32")
    if framework == "pytorch":
        import torch
        return torch.tensor(value)
    import tensorflow as tf
    return tf.constant(value)


def array(value):
    return value.detach().numpy() if hasattr(value, "detach") else value.numpy()


def assign(parameter, value):
    if hasattr(parameter, "assign"):
        parameter.assign(value)
    else:
        import torch
        with torch.no_grad():
            parameter.copy_(torch.tensor(value, dtype=parameter.dtype))


def linear(layer, weight, bias=None):
    weight = np.asarray(weight, "float32")
    assign(layer.kernel if hasattr(layer, "kernel") else layer.weight,
           weight.T if hasattr(layer, "kernel") else weight)
    if bias is not None:
        assign(layer.bias, bias)


def fill(model, value):
    parameters = model.parameters() if hasattr(model, "parameters") else model.trainable_variables
    for p in parameters:
        assign(p, np.full(tuple(p.shape), value, "float32"))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_partner_embeddings_high_orders_circular_geometry_and_dense_composition(framework):
    t = lambda x: tensor(framework, x)
    get = lambda name: cls(framework, "field_products", name)
    partner = t(np.arange(1, 10).reshape(1, 3, 3, 1))
    np.testing.assert_allclose(array(get("FieldAwareFactorization")(3, 1)(partner)), [[8, 21, 48]])
    np.testing.assert_allclose(array(get("OperationAwareInteraction")(3, 1)(partner)), [[1, 5, 9, 8, 21, 48]])
    x = t([[[1], [2], [3]]])
    np.testing.assert_allclose(array(get("HigherOrderFactorization")(3, 1, 3)(x)), [[6]])
    np.testing.assert_allclose(array(get("InteractionMachine")(3, 1, 3)(x)), [[6, 11, 6]])
    x = t([[[1, 2, 3], [4, 5, 6]]])
    np.testing.assert_allclose(array(get("HolographicInteraction")(2, 3)(x)), [[[31, 31, 28]]], atol=2e-5)
    np.testing.assert_allclose(array(get("HolographicInteraction")(2, 3, "correlation")(x)), [[[32, 29, 29]]], atol=2e-5)
    x = t([[[1, 0], [0, 1]]])
    np.testing.assert_allclose(array(get("LorentzInteraction")(2, 2)(x)), [[1.5 - np.sqrt(2)]], atol=1e-6)
    out = get("DenseSparseInteraction")(2, 2)(t([[3, 4]]), t([[[1, 0], [0, 2]]]))
    np.testing.assert_allclose(array(out), [[0, 3, 8, 3, 4]])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_log_orders_euler_polar_and_generalized_outer_products(framework):
    t = lambda x: tensor(framework, x)
    log = cls(framework, "field_algebra", "LogarithmicInteraction")(2, 2, 1)
    x = t([[[2, 3], [4, 5]]])
    log(x)
    assign(log.orders, np.ones((2, 1), "float32"))
    np.testing.assert_allclose(array(log(x)), [[[8, 15]]], atol=3e-6)
    euler = cls(framework, "field_algebra", "EulerInteraction")(2, 1, 1)
    real, imag = t([[[2], [3]]]), t([[[0], [0]]])
    euler(real, imag)
    fill(euler, 0)
    assign(euler.orders, np.ones((2, 1), "float32"))
    out_r, out_i = euler(real, imag)
    np.testing.assert_allclose(array(out_r), [[[6]]], atol=2e-6)
    np.testing.assert_array_equal(array(out_i), [[[0]]])
    aoa = cls(framework, "field_algebra", "GeneralizedInteraction")(2, 1, 2, 1)
    out = aoa(t([[[1, 2], [3, 4]]]), t([[[5, 6]]]))
    np.testing.assert_allclose(array(out), [[[20, 36]]])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_hierarchical_graph_decomposed_and_sam_equations(framework):
    t = lambda x: tensor(framework, x)
    get = lambda name: cls(framework, "field_attention", name)
    hierarchy = get("HierarchicalInteractionAggregation")(2, 1, 2)
    x = t([[[1], [2]]])
    hierarchy(x)
    fill(hierarchy, 0)
    np.testing.assert_allclose(array(hierarchy(x)), [[2.625]])
    graph = get("FieldGraphInteraction")(2, 1)
    graph(x)
    fill(graph, 1)
    output, weights = graph(x, return_attention=True)
    np.testing.assert_allclose(array(output), [[[3], [3]]])
    np.testing.assert_array_equal(array(weights), [[[0, 1], [1, 0]]])
    destine = get("DisentangledFieldAttention")(2, 2, 2)
    x = t([[[1, 2], [1, 2]]])
    destine(x)
    fill(destine, 0)
    for layer in (destine.query, destine.key, destine.value):
        linear(layer, np.eye(2), np.zeros(2))
    output, weights = destine(x, return_attention=True)
    np.testing.assert_allclose(array(output), [[[2, 4], [2, 4]]])
    np.testing.assert_allclose(array(weights).sum(-1), np.full((1, 2, 2), 2))
    sam = get("SelfAttentionInteraction")(2, 2)
    x = t([[[1, 2], [3, 4]]])
    np.testing.assert_allclose(array(sam(x)), [[[[5, 20], [33, 88]], [[33, 88], [225, 400]]]])
    for mode, expected in (("2A", [[[[5, 5], [11, 11]], [[11, 11], [25, 25]]]]),
                           ("3A", [[[16, 16], [36, 36]]]), ("3E", [[[38, 108], [258, 488]]])):
        sam = get("SelfAttentionInteraction")(2, 2, mode, residual=False)
        sam(x)
        if mode.startswith("3"):
            linear(sam.key, np.eye(2))
        np.testing.assert_allclose(array(sam(x)), expected)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_convolution_kmax_retains_field_order_and_recombination_generates_fields(framework):
    t = lambda x: tensor(framework, x)
    ccpm = cls(framework, "field_convolution", "OrderedKMaxConvolution")(3, 1, 1, 1, 2)
    x = t([[[3], [1], [2]]])
    ccpm(x)
    fill(ccpm, 0)
    assign(ccpm.conv.kernel if hasattr(ccpm.conv, "kernel") else ccpm.conv.weight,
           np.ones(tuple(ccpm.conv.kernel.shape if hasattr(ccpm.conv, "kernel") else ccpm.conv.weight.shape)))
    np.testing.assert_allclose(array(ccpm(x)), [[np.tanh(3), np.tanh(2)]], atol=1e-6)
    fg = cls(framework, "field_convolution", "ConvolutionalFieldGenerator")(4, 1, 1, 1, 2, 2)
    x = t([[[1], [3], [2], [4]]])
    fg(x)
    fill(fg, 0)
    assign(fg.conv.kernel if hasattr(fg.conv, "kernel") else fg.conv.weight,
           np.ones(tuple(fg.conv.kernel.shape if hasattr(fg.conv, "kernel") else fg.conv.weight.shape)))
    linear(fg.recombine, np.eye(2), np.zeros(2))
    np.testing.assert_allclose(array(fg(x)), [[[np.tanh(np.tanh(3))], [np.tanh(np.tanh(4))]]], atol=1e-6)


def inputs(framework, name, masked=True):
    rng = np.random.default_rng(74)
    valid = np.array([[True, False, True], [False, False, False]]) if masked else np.ones((2, 3), bool)
    x = rng.normal(.5, .3, (2, 3, 2)).astype("float32")
    x[~valid] = np.nan
    t = lambda v: tensor(framework, v)
    if name in {"FieldAwareFactorization", "OperationAwareInteraction"}:
        pair = np.repeat(x[:, :, None, :], 3, 2)
        return [t(pair), t(valid)]
    if name == "DenseSparseInteraction":
        return [t(np.ones((2, 2))), t(x), t(valid)]
    if name == "EulerInteraction":
        return [t(x), t(x * .3), t(valid)]
    if name == "GeneralizedInteraction":
        return [t(x), t(np.ones((2, 2, 2))), t(valid), t(np.ones((2, 2), bool))]
    return [t(x), t(valid)]


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_partner_padding_is_sanitized_on_both_field_axes(framework):
    valid = np.array([[True, False, True]])
    values = np.arange(1,10,dtype="float32").reshape(1,3,3,1)
    values[~(valid[:,:,None] & valid[:,None,:])] = np.nan
    for name, expected in (("FieldAwareFactorization", [[0,21,0]]),
                           ("OperationAwareInteraction", [[1,0,9,0,21,0]])):
        model = cls(framework,"field_products",name)(3,1)
        np.testing.assert_allclose(array(model(tensor(framework,values),tensor(framework,valid))),expected)


@pytest.mark.parametrize("module,name,args", SPECS)
def test_native_framework_parity_padding_gradients_and_standalone(module, name, args):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("tensorflow")
    pt, kt = [cls(fw, module, name)(*args) for fw in ("pytorch", "tensorflow")]
    px, kx = inputs("pytorch", name), inputs("tensorflow", name)
    kt(*kx)
    fill(pt, .2)
    fill(kt, .2)
    px[0].requires_grad_()
    kx[0] = tf.Variable(kx[0])
    out_pt = pt(*px)
    with tf.GradientTape() as tape:
        out_tf = kt(*kx)
        tf_values = out_tf if isinstance(out_tf, tuple) else (out_tf,)
        loss_tf = sum(tf.reduce_sum(tf.square(v)) for v in tf_values)
    pt_values = out_pt if isinstance(out_pt, tuple) else (out_pt,)
    sum(v.square().sum() for v in pt_values).backward()
    gradients = tape.gradient(loss_tf, [kx[0]] + kt.trainable_variables)
    for a, b in zip(pt_values, tf_values):
        assert np.isfinite(array(a)).all()
        np.testing.assert_allclose(array(a), array(b), atol=3e-5, rtol=2e-5)
        if name != "DenseSparseInteraction":
            np.testing.assert_array_equal(array(a)[1], np.zeros_like(array(a)[1]))
    assert px[0].grad is not None and torch.isfinite(px[0].grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pt.parameters())
    assert all(g is not None and np.isfinite(array(g)).all() for g in gradients)
    np.testing.assert_allclose(array(px[0].grad), array(gradients[0]), atol=1e-4, rtol=2e-4)
    for fw in ("pytorch", "tensorflow"):
        loaded = importlib.import_module(f"model_evo_harness.models.{fw}.{module}")
        namespace = {"__name__": "standalone_fields"}
        exec(compile(Path(loaded.__file__).read_text(), "standalone_fields.py", "exec"), namespace)
        standalone = namespace[name](*args)
        standalone(*inputs(fw, name))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_numeric_guards_invalid_dimensions_and_zero_polar_gradients(framework):
    t = lambda x: tensor(framework, x)
    log_cls = cls(framework, "field_algebra", "LogarithmicInteraction")
    log = log_cls(2, 1, 1)
    x = t([[[0], [1e20]]])
    log(x)
    assign(log.orders, np.full((2, 1), 1000, "float32"))
    assert np.isfinite(array(log(x))).all()
    for kwargs in ({"epsilon": 0}, {"max_log": float("inf")}, {"max_log": 1000}):
        with pytest.raises(ValueError):
            log_cls(2, 1, 1, **kwargs)
    with pytest.raises(ValueError):
        cls(framework, "field_products", "HigherOrderFactorization")(2, 1, 3)
    with pytest.raises(ValueError):
        cls(framework, "field_attention", "DisentangledFieldAttention")(2, 3, 2)
    euler = cls(framework, "field_algebra", "EulerInteraction")(2, 1, 1)
    if framework == "pytorch":
        import torch
        r = torch.zeros(1, 2, 1, requires_grad=True)
        a, b = euler(r, r)
        (a.square().sum() + b.square().sum()).backward()
        assert torch.isfinite(r.grad).all()
    else:
        import tensorflow as tf
        r = tf.Variable(tf.zeros((1, 2, 1)))
        with tf.GradientTape() as tape:
            a, b = euler(r, r)
            loss = tf.reduce_sum(a*a + b*b)
        assert np.isfinite(array(tape.gradient(loss, r))).all()
