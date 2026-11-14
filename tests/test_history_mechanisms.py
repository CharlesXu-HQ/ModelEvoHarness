"""Formula, mask and gradient contracts for independent history mechanisms."""
import importlib
from pathlib import Path

import numpy as np
import pytest


SPECS = [
    ("history_matching", "BSTHistoryEncoder", dict(dim=4, num_heads=2, max_len=16)),
    ("history_matching", "DMINInterestRefiner", dict(dim=4, num_heads=2, max_len=16)),
    ("history_matching", "DMRHistoryMatching", dict(dim=4, max_len=16)),
    ("history_retrieval", "SIMHistorySelector", dict(dim=4, topk=5)),
    ("history_retrieval", "ETAHistorySelector", dict(dim=4, topk=5, hash_bits=4)),
    ("history_retrieval", "SDIMCollisionPooling", dict(dim=4, num_hashes=2, hash_bits=2)),
    ("history_retrieval", "TWINHeadwiseSelector", dict(dim=4, num_heads=2, topk=5)),
    ("history_retrieval", "MIRRNHistoryRefiner", dict(dim=4, topk=5, hash_bits=4, num_blocks=2)),
]


def module(framework, name):
    path = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models" / framework / (name + ".py")
    assert path.is_file(), f"Missing mechanism implementation: {path}"
    return importlib.import_module(f"model_evo_harness.models.{framework}.{name}")


def tensor(framework, x):
    x = np.asarray(x)
    x = x if x.dtype == bool else x.astype(np.float32)
    if framework == "pytorch":
        import torch
        return torch.tensor(x)
    import tensorflow as tf
    return tf.constant(x)


def array(x):
    return x.detach().numpy() if hasattr(x, "detach") else x.numpy()


def assign(x, value):
    if hasattr(x, "assign"):
        x.assign(value)
    else:
        import torch
        with torch.no_grad():
            x.copy_(torch.as_tensor(value, dtype=x.dtype))


def linear(layer, matrix):
    if hasattr(layer, "kernel"):
        assign(layer.kernel, matrix)
        if layer.bias is not None:
            assign(layer.bias, np.zeros(layer.bias.shape))
    else:
        assign(layer.weight, np.asarray(matrix).T)
        if layer.bias is not None:
            assign(layer.bias, np.zeros(tuple(layer.bias.shape)))


def call(model, name, target, history, valid):
    if name == "MIRRNHistoryRefiner":
        return model(target, history, valid, recent_mask=valid)
    return model(target, history, valid)


def fixture(framework, length=3):
    rng = np.random.default_rng(91)
    valid = np.zeros((2, length), dtype=bool)
    valid[0, :min(2, length)] = True
    history = np.where(valid[..., None], rng.normal(size=(2, length, 4)), np.nan)
    return tuple(tensor(framework, x) for x in (rng.normal(size=(2, 4)), history, valid))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
@pytest.mark.parametrize("mod,name,kwargs", SPECS)
@pytest.mark.parametrize("length", [0, 3])
def test_padding_never_becomes_evidence_and_empty_is_zero(framework, mod, name, kwargs, length):
    model = getattr(module(framework, mod), name)(**kwargs)
    target, history, valid = fixture(framework, length)
    result = call(model, name, target, history, valid)
    for key, value in result.items():
        values = array(value)
        if values.dtype.kind == "f":
            assert np.isfinite(values).all(), (name, key)
    rep = array(result["representation"])
    np.testing.assert_array_equal(rep[1], 0)
    if length == 0:
        np.testing.assert_array_equal(rep, 0)
    if "selected_mask" in result:
        selected_mask = array(result["selected_mask"])
        indices = array(result["indices"])
        np.testing.assert_array_equal(indices[~selected_mask], -1)
        if length:
            rows = np.broadcast_to(np.arange(2).reshape((2,) + (1,)*(indices.ndim-1)), indices.shape)
            assert array(valid)[rows[selected_mask], indices[selected_mask]].all()


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_sim_preserves_negative_scores_in_coarse_path_and_masks_topk(framework):
    model = module(framework, "history_retrieval").SIMHistorySelector(2, topk=4)
    t = lambda x: tensor(framework, x)
    target, history, valid = t([[1, 0]]), t([[[-2, 1], [-1, 3], [np.nan, np.nan]]]), t([[True, True, False]])
    model(target, history, valid)
    linear(model.query, np.eye(2)); linear(model.key, np.eye(2))
    result = model(target, history, valid)
    np.testing.assert_allclose(array(result["coarse"]), [[5, -5]], atol=1e-6)
    np.testing.assert_array_equal(array(result["indices"]), [[1, 0, -1, -1]])
    np.testing.assert_array_equal(array(result["selected_mask"]), [[True, True, False, False]])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_eta_hamming_selection_and_sdim_collision_sum_have_distinct_results(framework):
    m = module(framework, "history_retrieval")
    t = lambda x: tensor(framework, x)
    args = (t([[1, 1]]), t([[[1, 2], [3, 4], [-1, 2], [-1, -1], [np.nan, np.nan]]]), t([[True, True, True, True, False]]))
    eta = m.ETAHistorySelector(2, topk=3, hash_bits=2)
    eta(*args); assign(eta.rotations, np.eye(2))
    result = eta(*args)
    np.testing.assert_array_equal(array(result["indices"]), [[0, 1, 2]])
    np.testing.assert_allclose(array(result["hash_distance"]), [[0, 0, 1, 2, 0]])
    sdim = m.SDIMCollisionPooling(2, num_hashes=1, hash_bits=2)
    sdim(*args); assign(sdim.rotations, np.eye(2)[None])
    result = sdim(*args)
    np.testing.assert_array_equal(array(result["collision_mask"]), [[[True, True, False, False, False]]])
    # Sum within each bucket, average across hashes, not mean per event.
    np.testing.assert_allclose(array(result["representation"]), [[4, 6]])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_twin_heads_select_different_events_and_report_original_indices(framework):
    model = module(framework, "history_retrieval").TWINHeadwiseSelector(4, num_heads=2, topk=1)
    t = lambda x: tensor(framework, x)
    args = (t([[1, 0, 1, 0]]), t([[[3, 0, 0, 0], [0, 0, 4, 0], [np.nan]*4]]), t([[True, True, False]]))
    model(*args)
    for layer in (model.query, model.key, model.value, model.out_proj):
        linear(layer, np.eye(4))
    result = model(*args)
    np.testing.assert_array_equal(array(result["indices"]), [[[0], [1]]])
    np.testing.assert_allclose(array(result["representation"]), [[3, 0, 4, 0]])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_bst_is_candidate_aware_and_dmin_preserves_causal_head_states(framework):
    m = module(framework, "history_matching")
    target, history, valid = fixture(framework)
    bst = m.BSTHistoryEncoder(4, num_heads=2, max_len=16)
    initial = bst(target, history, valid)
    changed_target = tensor(framework, array(target) + [2, -1, 3, 0])
    changed = bst(changed_target, history, valid)
    assert not np.allclose(array(initial["states"])[0, :2], array(changed["states"])[0, :2])
    dmin = m.DMINInterestRefiner(4, num_heads=2, max_len=16)
    initial = dmin(target, history, valid)
    altered = array(history); altered[0, 1] += 20
    changed = dmin(target, tensor(framework, altered), valid)
    np.testing.assert_allclose(array(initial["head_states"])[0, :, 0], array(changed["head_states"])[0, :, 0], atol=1e-6)
    assert array(initial["interests"]).shape == (2, 2, 4)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_dmr_user_path_is_target_independent_but_item_path_is_conditioned(framework):
    m = module(framework, "history_matching").DMRHistoryMatching(4, max_len=16)
    target, history, valid = fixture(framework)
    # Keep a known active target path; random ReLU gates can legitimately close it.
    linear(m.item_query, np.concatenate([np.eye(4), np.zeros((8, 4))]))
    linear(m.item_score1, np.concatenate([np.eye(4), np.eye(4), np.zeros((8, 4))]))
    linear(m.item_score2, np.ones((4, 1)))
    a = m(target, history, valid)
    b = m(tensor(framework, array(target) + [10, 0, -5, 1]), history, valid)
    np.testing.assert_allclose(array(a["user_states"]), array(b["user_states"]), atol=1e-6)
    assert not np.allclose(array(a["item_scores"])[0, :2], array(b["item_scores"])[0, :2])
    np.testing.assert_allclose(array(a["item_score"]), array(a["item_scores"]).sum(-1), atol=1e-6)


@pytest.mark.parametrize("mod,name,kwargs", SPECS)
def test_tensorflow_dynamic_graph_accepts_zero_and_variable_lengths(mod, name, kwargs):
    import tensorflow as tf
    model = getattr(module("tensorflow", mod), name)(**kwargs)
    @tf.function(autograph=False, input_signature=[tf.TensorSpec([None, 4], tf.float32), tf.TensorSpec([None, None, 4], tf.float32), tf.TensorSpec([None, None], tf.bool)])
    def run(target, history, valid):
        return call(model, name, target, history, valid)
    for length in (3, 0, 5):
        args = fixture("tensorflow", length)
        output = run(*args)
        assert np.isfinite(array(output["representation"])).all()
        np.testing.assert_array_equal(array(output["representation"])[1], 0)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
@pytest.mark.parametrize("mod,name,kwargs", SPECS)
def test_masked_nan_inputs_have_finite_gradients_and_trainable_paths(framework, mod, name, kwargs):
    model = getattr(module(framework, mod), name)(**kwargs)
    target, history, valid = fixture(framework)
    if framework == "pytorch":
        import torch
        history.requires_grad_(True)
        result = call(model, name, target, history, valid)
        # A weighted objective avoids the zero sum induced by LayerNorm.
        loss = sum((x * torch.arange(1, x.numel()+1, device=x.device).reshape(x.shape)).sum() for x in result.values() if x.is_floating_point() and x.requires_grad)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        input_gradient = history.grad
    else:
        import tensorflow as tf
        history = tf.Variable(history)
        with tf.GradientTape() as tape:
            result = call(model, name, target, history, valid)
            loss = sum(tf.reduce_sum(x * tf.reshape(tf.cast(tf.range(1, tf.size(x)+1), x.dtype), tf.shape(x))) for x in result.values() if x.dtype.is_floating)
        gradients_all = tape.gradient(loss, [history] + model.trainable_variables)
        input_gradient, gradients = gradients_all[0], gradients_all[1:]
    assert input_gradient is not None
    assert np.isfinite(array(input_gradient)).all()
    np.testing.assert_array_equal(array(input_gradient)[~array(valid)], 0)
    assert all(g is not None and np.isfinite(array(g)).all() for g in gradients)
    if gradients:
        assert any(np.any(array(g) != 0) for g in gradients)


def _map_weights(torch_model, tf_model):
    """Map actual public module paths; no framework defaults enter parity."""
    import torch
    def resolve(path):
        obj = tf_model
        for piece in path.split('.') if path else []:
            obj = obj[int(piece)] if piece.isdigit() else getattr(obj, piece)
        return obj
    for name, layer in torch_model.named_modules():
        mapped = resolve(name)
        if isinstance(layer, torch.nn.Linear):
            assign(mapped.kernel, array(layer.weight).T)
            if layer.bias is not None:
                assign(mapped.bias, array(layer.bias))
        elif isinstance(layer, torch.nn.LayerNorm):
            assign(mapped.gamma, array(layer.weight))
            assign(mapped.beta, array(layer.bias))
        for parameter, value in layer.named_parameters(recurse=False):
            if not isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm)):
                assign(getattr(mapped, parameter), array(value))
        for buffer, value in layer.named_buffers(recurse=False):
            assign(getattr(mapped, buffer), array(value))


@pytest.mark.parametrize("mod,name,kwargs", SPECS)
def test_mapped_weights_match_all_outputs_across_frameworks(mod, name, kwargs):
    import torch
    torch.manual_seed(182)
    pt = getattr(module("pytorch", mod), name)(**kwargs)
    tf_model = getattr(module("tensorflow", mod), name)(**kwargs)
    _map_weights(pt, tf_model)
    for length in (0, 3, 5):
        first = call(pt, name, *fixture("pytorch", length))
        second = call(tf_model, name, *fixture("tensorflow", length))
        assert first.keys() == second.keys()
        for key in first:
            np.testing.assert_allclose(array(first[key]), array(second[key]), atol=2e-5, rtol=2e-5, err_msg=f"{name}.{key}, L={length}")


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_mirrn_full_offdiagonal_channel_mixing_odd_length_and_mask(framework):
    model = module(framework, "history_retrieval").MIRRNHistoryRefiner(4, topk=3, hash_bits=2, num_blocks=2)
    t = lambda x: tensor(framework, x)
    real = np.zeros((3, 2, 2, 2), np.float32)
    real[0, 0, 0, 1] = 1  # channel 0 must contribute to output channel 1.
    assign(model.frequency_real, real)
    assign(model.frequency_imag, np.zeros_like(real))
    selected = t([[[1, 0, 0, 0], [0, 0, 0, 0], [np.nan]*4]])
    valid = t([[True, True, False]])
    result = array(model.refine(selected, valid, 0))
    # FFT + full real channel map gives residual [1,1,0,0] at first event.
    expected = 1 / np.sqrt(1 + 4e-5)
    np.testing.assert_allclose(result[0, 0], [expected, expected, -expected, -expected], atol=1e-6)
    np.testing.assert_allclose(result[0, 1:], 0, atol=1e-6)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_mirrn_missing_recent_view_has_no_phantom_interest(framework):
    model = module(framework, "history_retrieval").MIRRNHistoryRefiner(4, topk=3, hash_bits=4, num_blocks=2)
    target, history, valid = fixture(framework)
    result = model(target, history, valid, recent_mask=tensor(framework, np.zeros((2, 3), bool)))
    np.testing.assert_array_equal(array(result["interest_mask"]), [[True, False, True], [False, False, False]])
    np.testing.assert_array_equal(array(result["indices"])[:, 1], -1)
    np.testing.assert_array_equal(array(result["interests"])[:, 1], 0)
    np.testing.assert_array_equal(array(result["attention"])[:, 1], 0)
    selected_mask = array(result["selected_mask"])
    indices = array(result["indices"])
    for row, mask in zip(indices.reshape(-1, 3), selected_mask.reshape(-1, 3)):
        assert (np.diff(row[mask]) > 0).all()
    import tensorflow as tf
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        model(target, history, valid, recent_mask=tensor(framework, np.ones((2, 3), bool)))


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_arbitrarily_negative_valid_scores_still_beat_invalid_history(framework):
    model = module(framework, "history_retrieval").SIMHistorySelector(2, topk=1)
    t = lambda x: tensor(framework, x)
    args = (t([[1, 0]]), t([[[-2e10, 1], [np.nan, np.nan]]]), t([[True, False]]))
    model(*args)
    linear(model.query, np.eye(2)); linear(model.key, np.eye(2))
    result = model(*args)
    np.testing.assert_array_equal(array(result["indices"]), [[0]])
    np.testing.assert_array_equal(array(result["selected_mask"]), [[True]])
