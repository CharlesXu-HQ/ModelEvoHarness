import unittest

import numpy as np


def _array(value):
    return value.detach().numpy() if hasattr(value, "detach") else np.asarray(value)


class PaperSequenceTests(unittest.TestCase):
    def test_transact_valid_recent_and_empty_mask(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.paper_sequence import TransAct as PT
        from model_evo_harness.models.tensorflow.paper_sequence import TransAct as TF

        history = np.random.default_rng(1).normal(size=(2, 4, 4)).astype("float32")
        mask = np.array([[True, False, True, False], [False] * 4])
        changed = history.copy()
        changed[~mask] = 9999
        candidate = np.random.default_rng(5).normal(size=(2, 4)).astype("float32")
        for model, tensor, boolean in ((PT(4, 2, 3), torch.tensor, torch.tensor),
                                       (TF(4, 2, 3), tf.constant, tf.constant)):
            first = model(tensor(history), tensor(candidate), boolean(mask))
            second = model(tensor(changed), tensor(candidate), boolean(mask))
            self.assertEqual(tuple(first.shape), (2, 16))
            self.assertTrue(np.allclose(_array(first), _array(second), atol=1e-5))
            self.assertTrue(np.allclose(_array(first)[1], 0))
            self.assertTrue(np.allclose(_array(first)[0, 8:12], 0))
            other = model(tensor(history), tensor(candidate + 10), boolean(mask))
            self.assertFalse(np.allclose(_array(first)[0], _array(other)[0]))

    def test_pinnerformer_causality_and_dense_loss(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.paper_sequence import PinnerFormer as PT
        from model_evo_harness.models.tensorflow.paper_sequence import PinnerFormer as TF

        rng = np.random.default_rng(2)
        history = rng.normal(size=(2, 4, 4)).astype("float32")
        mask = np.array([[True, True, True, False], [False] * 4])
        changed = history.copy()
        changed[:, 2:] += 100
        for model, tensor, boolean in ((PT(4, 2, 4), torch.tensor, torch.tensor),
                                       (TF(4, 2, 4), tf.constant, tf.constant)):
            first = model.encode(tensor(history), boolean(mask))
            second = model.encode(tensor(changed), boolean(mask))
            self.assertTrue(np.allclose(_array(first)[:, :2], _array(second)[:, :2],
                                        atol=1e-5))
            self.assertTrue(np.allclose(_array(first)[1], 0))
            self.assertTrue(np.allclose(np.linalg.norm(_array(first)[0, :3], axis=-1), 1,
                                        atol=1e-5))
            self.assertTrue(np.allclose(np.linalg.norm(_array(model(
                tensor(history), boolean(mask)))[0]), 1, atol=1e-5))
        states = torch.randn(2, 4, 4, requires_grad=True)
        positive = torch.randn(2, 4, 4)
        negative = torch.randn(2, 4, 1, 4)
        supervision = torch.tensor([[True, True, False, False], [False] * 4])
        loss = PT.dense_all_action_loss(states, positive, negative, supervision)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(states.grad[0, :2].abs().sum()), 0)
        self.assertEqual(float(states.grad[1].abs().sum()), 0)
        variable = tf.Variable(states.detach().numpy())
        with tf.GradientTape() as tape:
            value = TF.dense_all_action_loss(variable, tf.constant(positive.numpy()),
                tf.constant(negative.numpy()), tf.constant(supervision.numpy()))
        grad = tape.gradient(value, variable)
        self.assertTrue(np.isfinite(value.numpy()))
        self.assertGreater(float(tf.reduce_sum(tf.abs(grad[0, :2]))), 0)
        self.assertEqual(float(tf.reduce_sum(tf.abs(grad[1]))), 0)

    def test_seq2slate_no_repeat_and_eligibility(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.paper_sequence import Seq2Slate as PT
        from model_evo_harness.models.tensorflow.paper_sequence import Seq2Slate as TF

        candidates = np.random.default_rng(3).normal(size=(2, 3, 4)).astype("float32")
        eligible = np.array([[True, False, True], [False, False, False]])
        for model, tensor, boolean in ((PT(4, 3), torch.tensor, torch.tensor),
                                       (TF(4, 3), tf.constant, tf.constant)):
            result = _array(model(tensor(candidates), boolean(eligible)))
            self.assertEqual(set(result[0, :2]), {0, 2})
            self.assertEqual(result[0, 2], -1)
            self.assertTrue(np.all(result[1] == -1))
            logits = _array(model.teacher_forced_logits(tensor(candidates[:1]),
                boolean(eligible[:1]), tensor(np.array([[0, 2]], np.int32))))
            self.assertEqual(logits.shape, (1, 2, 3))
            self.assertLess(float(logits[0, 0, 1]), -1e8)
            self.assertLess(float(logits[0, 1, 0]), -1e8)
        pt = PT(4, 2)
        with self.assertRaisesRegex(ValueError, "repeated"):
            pt.teacher_forced_logits(torch.tensor(candidates[:1]),
                                     torch.tensor(eligible[:1]), torch.tensor([[0, 0]]))

    def test_sequence_modules_propagate_gradients(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.paper_sequence import (
            TransAct as PTTrans, PinnerFormer as PTPinner, Seq2Slate as PTSlate)
        from model_evo_harness.models.tensorflow.paper_sequence import (
            TransAct as TFTrans, PinnerFormer as TFPinner, Seq2Slate as TFSlate)

        mask = np.array([[True, True, False]])
        eligible = np.array([[True, True, False]])
        base = np.random.default_rng(4).normal(size=(1, 3, 4)).astype("float32")
        source = torch.tensor(base, requires_grad=True)
        pt_loss = (PTTrans(4, 2, 2)(source, source[:, 0], torch.tensor(mask)).sum() +
                   PTPinner(4, 2, 3)(source, torch.tensor(mask)).sum() +
                   PTSlate(4, 2).teacher_forced_logits(
                       source, torch.tensor(eligible), torch.tensor([[0, 1]]))
                   [:, :, :2].sum())
        pt_loss.backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertGreater(float(source.grad.abs().sum()), 0)
        variable = tf.Variable(base)
        with tf.GradientTape() as tape:
            tf_loss = (tf.reduce_sum(TFTrans(4, 2, 2)(variable, variable[:, 0],
                                                     tf.constant(mask))) +
                       tf.reduce_sum(TFPinner(4, 2, 3)(variable, tf.constant(mask))) +
                       tf.reduce_sum(TFSlate(4, 2).teacher_forced_logits(
                           variable, tf.constant(eligible), tf.constant([[0, 1]]))[:, :, :2]))
        gradient = tape.gradient(tf_loss, variable)
        self.assertTrue(np.isfinite(gradient.numpy()).all())
        self.assertGreater(float(tf.reduce_sum(tf.abs(gradient))), 0)

    def test_seq2slate_first_step_learns_from_candidate_set_and_float16_mask(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.paper_sequence import Seq2Slate as PT
        from model_evo_harness.models.tensorflow.paper_sequence import Seq2Slate as TF

        values = np.array([[[1., 0.], [0., 2.], [5., 5.]]], np.float32)
        changed = values.copy()
        changed[0, 1] = [2., 0.]
        allowed = np.array([[True, True, False]])
        targets = np.array([[0]], np.int32)
        pt = PT(2, 1).half()
        x = torch.tensor(values, dtype=torch.float16, requires_grad=True)
        logits = pt.teacher_forced_logits(x, torch.tensor(allowed), torch.tensor(targets))
        other = pt.teacher_forced_logits(torch.tensor(changed, dtype=torch.float16),
                                          torch.tensor(allowed), torch.tensor(targets))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(torch.allclose(logits[0, 0, :2], other[0, 0, :2]))
        logits[0, 0, 0].backward()
        self.assertGreater(float(x.grad[0, 1].abs().sum()), 0)
        tf_model = TF(2, 1)
        first = tf_model.teacher_forced_logits(tf.constant(values, tf.float16),
                                                tf.constant(allowed), tf.constant(targets))
        second = tf_model.teacher_forced_logits(tf.constant(changed, tf.float16),
                                                 tf.constant(allowed), tf.constant(targets))
        self.assertTrue(np.isfinite(first.numpy()).all())
        self.assertFalse(np.allclose(first.numpy()[0, 0, :2], second.numpy()[0, 0, :2]))
        variable = tf.Variable(values)
        with tf.GradientTape() as tape:
            first_score = tf_model.teacher_forced_logits(variable,
                tf.constant(allowed), tf.constant(targets))[0, 0, 0]
        gradient = tape.gradient(first_score, variable)
        self.assertGreater(float(tf.reduce_sum(tf.abs(gradient[0, 1]))), 0)

    def test_tensorflow_dynamic_sequence_lengths_in_graph(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.paper_sequence import (
            TransAct, PinnerFormer, Seq2Slate)

        transact = TransAct(4, 2, 2)
        pinner = PinnerFormer(4, 2, 8)
        slate = Seq2Slate(4, 3)

        @tf.function(input_signature=(tf.TensorSpec((None, None, 4), tf.float32),
                                      tf.TensorSpec((None, 4), tf.float32),
                                      tf.TensorSpec((None, None), tf.bool)))
        def encode(history, candidate, mask):
            return transact(history, candidate, mask), pinner.encode(history, mask)

        @tf.function(input_signature=(tf.TensorSpec((None, None, 4), tf.float32),
                                      tf.TensorSpec((None, None), tf.bool),
                                      tf.TensorSpec((None, None), tf.int32)))
        def decode(candidates, eligible, targets):
            return slate.teacher_forced_logits(candidates, eligible, targets), slate(
                candidates, eligible)

        for length in (3, 5):
            values = tf.random.normal((1, length, 4))
            mask = tf.concat((tf.ones((1, length - 1), tf.bool),
                              tf.zeros((1, 1), tf.bool)), 1)
            feature, states = encode(values, tf.ones((1, 4)), mask)
            self.assertEqual(tuple(feature.shape), (1, 12))
            self.assertEqual(tuple(states.shape), (1, length, 4))
            targets = tf.constant([[0, 1]]) if length == 3 else tf.constant([[0, 1, 2]])
            logits, selected = decode(values, tf.ones((1, length), tf.bool), targets)
            self.assertEqual(tuple(logits.shape), (1, int(targets.shape[1]), length))
            self.assertEqual(tuple(selected.shape), (1, 3))
