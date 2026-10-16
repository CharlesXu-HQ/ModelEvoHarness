"""Regression evidence for issues found during independent review."""
import numpy as np
import torch
import tensorflow as tf

from model_evo_harness.models.pytorch import paper_ranking as pt_rank
from model_evo_harness.models.tensorflow import paper_ranking as tf_rank
from model_evo_harness.models.pytorch import paper_sequence as pt_seq
from model_evo_harness.models.tensorflow import paper_sequence as tf_seq


def test_rank_losses_ignore_nonfinite_padding_and_keep_finite_gradients():
    for name in ("RankNet", "LambdaRank"):
        pt, kt = getattr(pt_rank, name)(), getattr(tf_rank, name)()
        scores = torch.tensor([[2., 1., -float("inf")]], requires_grad=True)
        labels, mask = torch.tensor([[2., 1., float("nan")]]), torch.tensor([[True, True, False]])
        value = pt(scores, labels, mask)
        reference = pt(scores[:, :2], labels[:, :2], mask[:, :2])
        torch.testing.assert_close(value, reference)
        value.backward()
        assert torch.isfinite(scores.grad).all() and scores.grad[0, 2] == 0
        variable = tf.Variable([[2., 1., -float("inf")]])
        with tf.GradientTape() as tape:
            tf_value = kt(variable, tf.constant([[2., 1., float("nan")]]), tf.constant(mask.numpy()))
        grad = tape.gradient(tf_value, variable)
        np.testing.assert_allclose(tf_value.numpy(), value.detach().numpy(), rtol=1e-6)
        assert np.isfinite(grad.numpy()).all() and grad.numpy()[0, 2] == 0
        # Relevance labels commonly arrive as integers.
        np.testing.assert_allclose(kt(tf.constant([[2., 1.]]), tf.constant([[2, 1]]),
                                     tf.constant([[True, True]])).numpy(), value.detach().numpy(), rtol=1e-6)


def test_empty_pointer_candidates_and_teacher_targets_in_tf_graph():
    pt, kt = pt_seq.Seq2Slate(4, 2), tf_seq.Seq2Slate(4, 2)
    assert pt(torch.empty((1, 0, 4)), torch.empty((1, 0), dtype=torch.bool)).tolist() == [[-1, -1]]
    assert pt.teacher_forced_logits(torch.ones((1, 2, 4)), torch.ones((1, 2), dtype=torch.bool),
                                    torch.empty((1, 0), dtype=torch.long)).shape == (1, 0, 2)
    compiled = tf.function(kt.call, input_signature=[tf.TensorSpec((None, None, 4), tf.float32),
                                                   tf.TensorSpec((None, None), tf.bool)])
    np.testing.assert_array_equal(compiled(tf.zeros((1, 0, 4)), tf.zeros((1, 0), tf.bool)), [[-1, -1]])
    teacher = tf.function(kt.teacher_forced_logits, input_signature=[
        tf.TensorSpec((None, None, 4), tf.float32), tf.TensorSpec((None, None), tf.bool),
        tf.TensorSpec((None, None), tf.int32)])
    assert teacher(tf.ones((1, 2, 4)), tf.ones((1, 2), tf.bool), tf.zeros((1, 0), tf.int32)).shape == (1, 0, 2)


def test_transact_same_weights_match_across_frameworks():
    torch.manual_seed(11)
    dim, heads = 4, 2
    pt, kt = pt_seq.TransAct(dim, heads, 2), tf_seq.TransAct(dim, heads, 2)
    history = np.random.default_rng(5).normal(size=(1, 3, dim)).astype("float32")
    candidate = np.ones((1, dim), dtype="float32")
    mask = np.array([[True, True, False]])
    kt(tf.constant(history), tf.constant(candidate), tf.constant(mask))
    arr = lambda x: x.detach().numpy()
    kt.fuse.set_weights([arr(pt.fuse.weight).T, arr(pt.fuse.bias)])
    weights = []
    for w, b in zip(pt.encoder.self_attn.in_proj_weight.chunk(3), pt.encoder.self_attn.in_proj_bias.chunk(3)):
        weights.extend([arr(w).T.reshape(dim, heads, dim//heads), arr(b).reshape(heads, dim//heads)])
    weights.extend([arr(pt.encoder.self_attn.out_proj.weight).T.reshape(heads, dim//heads, dim),
                    arr(pt.encoder.self_attn.out_proj.bias)])
    kt.encoder.attention.set_weights(weights)
    for target, source in ((kt.encoder.norm1, pt.encoder.norm1), (kt.encoder.norm2, pt.encoder.norm2)):
        target.set_weights([arr(source.weight), arr(source.bias)])
    for target, source in ((kt.encoder.feedforward.layers[0], pt.encoder.linear1),
                            (kt.encoder.feedforward.layers[1], pt.encoder.linear2)):
        target.set_weights([arr(source.weight).T, arr(source.bias)])
    expected = arr(pt(torch.tensor(history), torch.tensor(candidate), torch.tensor(mask)))
    actual = kt(tf.constant(history), tf.constant(candidate), tf.constant(mask)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_dpp_fixed_capacity_includes_subunit_positive_variance():
    # Fixed-capacity selection continues for a non-singular kernel even if log-det decreases.
    assert pt_rank.DPPGreedy(2)(torch.tensor([[[.5, 0.], [0., .25]]]), torch.tensor([[True, True]])).tolist() == [[0, 1]]
    np.testing.assert_array_equal(tf_rank.DPPGreedy(2)(tf.constant([[[.5, 0.], [0., .25]]]),
                                                       tf.constant([[True, True]])).numpy(), [[0, 1]])
