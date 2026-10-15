"""Numerical contracts for isolated cold-start, bias and multimodal heads."""
import importlib
import math

import numpy as np
import pytest


@pytest.fixture(params=["pytorch", "tensorflow"])
def backend(request):
    name = request.param
    runtime = pytest.importorskip("torch" if name == "pytorch" else "tensorflow")
    return name, runtime


def tensor(backend, values, dtype="float"):
    name, lib = backend
    if name == "pytorch":
        return lib.tensor(values, dtype={"float": lib.float32, "int": lib.long, "bool": lib.bool}[dtype])
    return lib.constant(values, dtype={"float": lib.float32, "int": lib.int32, "bool": lib.bool}[dtype])


def array(value):
    return value.detach().numpy() if hasattr(value, "detach") else value.numpy()


def module(backend, name):
    return importlib.import_module(f"model_evo_harness.models.{backend[0]}.{name}")


def test_pal_product_and_stable_joint_bce(backend):
    model = module(backend, "debias").PAL(2)
    logits = tensor(backend, [math.log(4), -1000., 1000.])
    positions = tensor(backend, [0, 1, 0], "int")
    outputs = model(logits, positions)
    np.testing.assert_allclose(array(outputs["observed"]), [0.4, 0., 0.5], atol=1e-6)
    value = model.objective(logits, positions, tensor(backend, [1., 1., 0.]))
    expected = (-math.log(.4) + 1000 + math.log(2) + math.log(2)) / 3
    np.testing.assert_allclose(array(value), expected, rtol=1e-6)
    if backend[0] == "tensorflow":
        tf = backend[1]
        compiled = tf.function(model.objective)
        np.testing.assert_allclose(array(compiled(logits, positions, tf.constant([1., 1., 0.]))), expected, rtol=1e-6)


def test_pal_gradients_and_position_free_serving(backend):
    name, lib = backend
    model = module(backend, "debias").PAL(2)
    positions, labels = tensor(backend, [0, 1], "int"), tensor(backend, [1., 0.])
    logits = tensor(backend, [0.3, -0.4])
    if name == "pytorch":
        logits.requires_grad_()
        loss = model.objective(logits, positions, labels)
        loss.backward()
        gradients = [logits.grad, model.examination.weight.grad]
    else:
        logits = lib.Variable(logits)
        with lib.GradientTape() as tape:
            loss = model.objective(logits, positions, labels)
        gradients = tape.gradient(loss, [logits] + model.trainable_variables)
    assert all(g is not None and np.isfinite(array(lib.convert_to_tensor(g) if name == "tensorflow" else g)).all() for g in gradients)
    np.testing.assert_allclose(array(model.relevance(logits)), 1 / (1 + np.exp(-array(logits))), atol=1e-6)


def test_meta_embedding_inner_update_has_outer_gradient(backend):
    name, lib = backend
    model = module(backend, "debias").MetaEmbedding(2, 2)
    attributes = tensor(backend, [[.2, .4]])
    if name == "pytorch":
        initial = model(attributes)
        cold = (initial - .3).square().sum()
        gradient, = lib.autograd.grad(cold, initial, create_graph=True)
        adapted = model.adapt(initial, gradient, .1)
        warm = (adapted + .2).square().sum()
        model.meta_loss(cold, warm).backward()
        outer = model.generator.weight.grad
    else:
        with lib.GradientTape() as outer_tape:
            with lib.GradientTape() as inner_tape:
                initial = model(attributes)
                inner_tape.watch(initial)
                cold = lib.reduce_sum(lib.square(initial - .3))
            gradient = inner_tape.gradient(cold, initial)
            adapted = model.adapt(initial, gradient, .1)
            warm = lib.reduce_sum(lib.square(adapted + .2))
            loss = model.meta_loss(cold, warm)
        outer = outer_tape.gradient(loss, model.trainable_variables)[0]
    np.testing.assert_allclose(array(adapted), array(initial) - .1 * 2 * (array(initial) - .3), atol=1e-6)
    assert np.isfinite(array(outer)).all() and np.abs(array(outer)).sum() > 0


def test_dice_mask_direction_and_discrepancy(backend):
    name, lib = backend
    model = module(backend, "debias").DICE(1, 2, 1)
    u, p, n = (tensor(backend, x, "int") for x in ([0], [0], [1]))
    model(u, p)
    tables = ((model.user_interest, [[1.]]), (model.user_conformity, [[2.]]),
              (model.item_interest, [[3.], [1.]]), (model.item_conformity, [[1.], [2.]]))
    for table, values in tables:
        if name == "pytorch":
            with lib.no_grad():
                table.weight.copy_(tensor(backend, values))
        else:
            table.set_weights([np.array(values, dtype="float32")])
    scores = model(u, p)
    np.testing.assert_allclose(array(scores["total"]), [5.])
    # Interest difference=+2, conformity difference=-2, total difference=0.
    for mask in (True, False):
        expected = math.log(2) + (0.2 * np.logaddexp(0, -2) if mask else 0.1 * np.logaddexp(0, 2))
        loss = model.objective(u, p, n, tensor(backend, [mask], "bool"))
        np.testing.assert_allclose(array(loss), expected, rtol=1e-6)
    penalty = model.objective(u, p, n, tensor(backend, [True], "bool"), discrepancy_weight=.2)
    # user discrepancy=1, item discrepancy=(4+1)/2=2.5
    np.testing.assert_allclose(array(penalty), math.log(2) + .2*np.logaddexp(0, -2) - .2*3.5, atol=1e-6)


def test_mmrec_four_streams_against_numpy_and_empty_history(backend):
    model = module(backend, "multimodal").MMRec()
    ht = np.array([[[1., 0.], [0., 2.], [1000., 1000.]]], dtype="float32")
    hi = np.array([[[0., 1.], [2., 0.], [-1000., 1000.]]], dtype="float32")
    ct, ci = np.array([[1., .5]], dtype="float32"), np.array([[.3, 1.]], dtype="float32")
    expected = np.zeros((1, 2))
    for h in (ht, hi):
        for c in (ct, ci):
            logits = (h[:, :2] * c[:, None]).sum(-1)
            weights = np.exp(logits - logits.max(-1, keepdims=True))
            weights /= weights.sum(-1, keepdims=True)
            expected += (weights[..., None] * h[:, :2]).sum(1)
    inputs = [tensor(backend, a) for a in (ht, hi, ct, ci)]
    mask = tensor(backend, [[True, True, False]], "bool")
    np.testing.assert_allclose(array(model.representation(*inputs, mask)), expected, rtol=1e-6)
    np.testing.assert_allclose(array(model(*inputs, mask)), (expected*(ct+ci)).sum(-1), rtol=1e-6)
    np.testing.assert_allclose(array(model(*inputs, tensor(backend, [[False]*3], "bool"))), 0)


def test_half_precision_attention_with_empty_and_integer_mask(backend):
    name, lib = backend
    logits = tensor(backend, [[2., 3.], [2., 3.]])
    logits = logits.half() if name == "pytorch" else lib.cast(logits, lib.float16)
    result = module(backend, "multimodal")._attention(logits, tensor(backend, [[0, 0], [1, 0]], "int"))
    np.testing.assert_allclose(array(result), [[0., 0.], [1., 0.]])


def test_dicm_two_channels_padding_and_gradient(backend):
    name, lib = backend
    model = module(backend, "multimodal").DICM(2, 2)
    history = tensor(backend, [[[1., 2.], [3., 1.], [1000., -1000.]]])
    query = tensor(backend, [[.2, .3]])
    mask = tensor(backend, [[True, True, False]], "bool")
    output = model(history, query, query, mask)
    changed = model(history, query, query * 10, mask)
    np.testing.assert_allclose(array(output)[:, :2], array(changed)[:, :2], atol=1e-6)
    empty = model(history, query, query, tensor(backend, [[False]*3], "bool"))
    np.testing.assert_allclose(array(empty), 0)
    if name == "pytorch":
        output.sum().backward()
        gradients = [p.grad for p in model.parameters()]
    else:
        with lib.GradientTape() as tape:
            loss = lib.reduce_sum(model(history, query, query, mask))
        gradients = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None and np.isfinite(array(g)).all() for g in gradients)
    assert any(np.abs(array(g)).sum() > 0 for g in gradients)


def test_distillation_zero_kl_and_detached_teacher(backend):
    name, lib = backend
    model = module(backend, "multimodal").DMTLDistillation()
    a = tensor(backend, [math.log(4)])  # p(click)=.8
    b = tensor(backend, [math.log(.3/.7)])  # p(long|click)=.3
    s = tensor(backend, [math.log(.24/.76)])
    np.testing.assert_allclose(array(model(a, b, s)), 0, atol=1e-6)
    if name == "pytorch":
        a.requires_grad_(); b.requires_grad_(); s = lib.tensor([2.], requires_grad=True)
        loss = model(a, b, s); loss.backward()
        assert a.grad is None and b.grad is None
        gradient = s.grad
    else:
        a, b, s = lib.Variable(a), lib.Variable(b), lib.Variable([2.])
        with lib.GradientTape() as tape:
            loss = model(a, b, s)
        ga, gb, gradient = tape.gradient(loss, [a, b, s])
        assert ga is None and gb is None
    assert float(array(loss)) > 0 and np.abs(array(gradient)).sum() > 0
    extreme = model(tensor(backend, [1000., -1000.]), tensor(backend, [1000., 1000.]), tensor(backend, [-1000., 1000.]))
    assert np.isfinite(array(extreme)).all()
    teacher = model.teacher_loss(tensor(backend, [math.log(4)]), b, tensor(backend, [1.]), tensor(backend, [1.]))
    np.testing.assert_allclose(array(teacher), -math.log(.8)-math.log(.24), atol=1e-6)
