import unittest

import numpy as np


class SequenceModelTests(unittest.TestCase):
    def test_pytorch_sequence_families_forward_and_backward(self):
        import torch
        from model_evo_harness.models.pytorch.sequence import (
            DIEN, DSIN, HSTU, MIND, NARM, SASRec, SDM)

        history = torch.randn(2, 4, 4, requires_grad=True)
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        target = torch.randn(2, 4)
        outputs = [
            MIND(4, 2, 2)(history, mask).flatten(1),
            SASRec(4, 2, 4)(history, mask),
            SDM(4, 2)(history, mask),
            NARM(4)(history, mask),
            HSTU(4)(history, torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]), mask)
                .flatten(1),
            DIEN(4, 8)(target, history, mask).unsqueeze(-1),
            DSIN(4, 8, 2)(target, history.reshape(2, 2, 2, 4),
                           mask.reshape(2, 2, 2)).unsqueeze(-1),
        ]
        for output in outputs:
            self.assertEqual(output.shape[0], 2)
            self.assertTrue(torch.isfinite(output).all())
            output.sum().backward(retain_graph=True)
        self.assertIsNotNone(history.grad)
        self.assertTrue(torch.isfinite(history.grad).all())

    def test_tensorflow_sequence_families_forward_and_gradient(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.sequence import (
            DIEN, DSIN, HSTU, MIND, NARM, SASRec, SDM)

        history = tf.Variable(tf.random.normal((2, 4, 4)))
        mask = tf.constant([[True, True, True, False], [True, True, False, False]])
        target = tf.random.normal((2, 4))
        with tf.GradientTape() as tape:
            outputs = [
                MIND(4, 2, 2)(history, mask),
                SASRec(4, 2, 4)(history, mask),
                SDM(4, 2)(history, mask),
                NARM(4)(history, mask),
                HSTU(4)(history, tf.constant([[1, 2, 3, 4], [1, 2, 3, 4]]), mask),
                DIEN(4, 8)(target, history, mask),
                DSIN(4, 8, 2)(target, tf.reshape(history, (2, 2, 2, 4)),
                               tf.reshape(mask, (2, 2, 2))),
            ]
            total = tf.add_n([tf.reduce_sum(output) for output in outputs])
        gradient = tape.gradient(total, history)
        self.assertIsNotNone(gradient)
        self.assertTrue(np.isfinite(gradient.numpy()).all())
        for output in outputs:
            self.assertEqual(int(output.shape[0]), 2)
            self.assertTrue(np.isfinite(output.numpy()).all())

    def test_pytorch_mask_contract_and_empty_padding_invariance(self):
        import torch
        from model_evo_harness.models.pytorch.sequence import NARM, SDM, _last

        values = torch.tensor([[[1.0], [2.0], [3.0]], [[7.0], [8.0], [9.0]]])
        mask = torch.tensor([[True, False, True], [False, False, False]])
        self.assertEqual(_last(values, mask).tolist(), [[3.0], [0.0]])
        empty = torch.zeros((2, 3), dtype=torch.bool)
        first = torch.randn(2, 3, 4)
        changed = first + 1000
        for model in (NARM(4), SDM(4, 2)):
            with self.subTest(model=type(model).__name__):
                self.assertTrue(torch.allclose(model(first, empty), model(changed, empty)))
                self.assertTrue(torch.allclose(model(first, empty), torch.zeros(2, 4)))
                with self.assertRaisesRegex(ValueError, "prefix"):
                    model(first, mask)

    def test_tensorflow_mask_contract_and_empty_padding_invariance(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.sequence import NARM, SDM, _last

        values = tf.constant([[[1.0], [2.0], [3.0]], [[7.0], [8.0], [9.0]]])
        mask = tf.constant([[True, False, True], [False, False, False]])
        self.assertEqual(_last(values, mask).numpy().tolist(), [[3.0], [0.0]])
        empty = tf.zeros((2, 3), dtype=tf.bool)
        first = tf.random.normal((2, 3, 4))
        changed = first + 1000
        for model in (NARM(4), SDM(4, 2)):
            with self.subTest(model=type(model).__name__):
                self.assertTrue(np.allclose(model(first, empty), model(changed, empty)))
                self.assertTrue(np.allclose(model(first, empty), np.zeros((2, 4))))
                with self.assertRaisesRegex((ValueError, tf.errors.InvalidArgumentError),
                                            "prefix"):
                    model(first, mask)

    def test_sdm_separate_histories_and_user_attention(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.sequence import SDM as TorchSDM
        from model_evo_harness.models.tensorflow.sequence import SDM as TensorFlowSDM

        short = torch.randn(2, 3, 4, requires_grad=True)
        long = torch.randn(2, 5, 4, requires_grad=True)
        user = torch.randn(2, 4, requires_grad=True)
        short_mask = torch.tensor([[True, True, False], [True, True, True]])
        long_mask = torch.tensor([[True, True, True, False, False],
                                  [True, True, True, True, False]])
        model = TorchSDM(4, 2)
        result = model(short, short_mask, long, long_mask, user)
        self.assertEqual(tuple(result.shape), (2, 4))
        self.assertIsInstance(model.short_lstm, torch.nn.LSTM)
        self.assertIsInstance(model.short_attention, torch.nn.MultiheadAttention)
        result.sum().backward()
        for source in (short, long, user):
            self.assertIsNotNone(source.grad)
            self.assertGreater(float(source.grad.abs().sum()), 0)

        tf_short = tf.Variable(short.detach().numpy())
        tf_long = tf.Variable(long.detach().numpy())
        tf_user = tf.Variable(user.detach().numpy())
        tf_model = TensorFlowSDM(4, 2)
        with tf.GradientTape() as tape:
            output = tf_model(tf_short, tf.constant(short_mask.numpy()), tf_long,
                              tf.constant(long_mask.numpy()), tf_user)
            objective = tf.reduce_sum(output)
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertIsInstance(tf_model.short_lstm, tf.keras.layers.LSTM)
        self.assertIsInstance(tf_model.short_attention,
                              tf.keras.layers.MultiHeadAttention)
        for gradient in tape.gradient(objective, [tf_short, tf_long, tf_user]):
            self.assertIsNotNone(gradient)
            self.assertGreater(float(tf.reduce_sum(tf.abs(gradient))), 0)

    def test_mind_target_aware_training_readout(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.sequence import MIND as TorchMIND
        from model_evo_harness.models.tensorflow.sequence import MIND as TensorFlowMIND

        interests = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
        right = torch.tensor([[1.0, 0.0]])
        left = -right
        torch_right = TorchMIND.label_aware_weights(interests, right)
        torch_left = TorchMIND.label_aware_weights(interests, left)
        self.assertGreater(float(torch_right[0, 0]), float(torch_right[0, 1]))
        self.assertGreater(float(torch_left[0, 1]), float(torch_left[0, 0]))
        history = torch.randn(2, 3, 4, requires_grad=True)
        result = TorchMIND(4, 2).training_readout(
            history, torch.ones(2, 3, dtype=torch.bool), torch.randn(2, 4))
        self.assertEqual(tuple(result.shape), (2, 4))
        result.sum().backward()
        self.assertGreater(float(history.grad.abs().sum()), 0)

        tf_interests = tf.constant(interests.numpy())
        tf_right = TensorFlowMIND.label_aware_weights(tf_interests,
                                                       tf.constant(right.numpy()))
        tf_left = TensorFlowMIND.label_aware_weights(tf_interests,
                                                      tf.constant(left.numpy()))
        self.assertGreater(float(tf_right[0, 0]), float(tf_right[0, 1]))
        self.assertGreater(float(tf_left[0, 1]), float(tf_left[0, 0]))
        tf_history = tf.Variable(history.detach().numpy())
        with tf.GradientTape() as tape:
            tf_result = TensorFlowMIND(4, 2).training_readout(
                tf_history, tf.ones((2, 3), dtype=tf.bool), tf.random.normal((2, 4)))
            objective = tf.reduce_sum(tf_result)
        self.assertEqual(tuple(tf_result.shape), (2, 4))
        gradient = tape.gradient(objective, tf_history)
        self.assertIsNotNone(gradient)
        self.assertGreater(float(tf.reduce_sum(tf.abs(gradient))), 0)

    def test_dsin_uses_backward_session_evolution(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.sequence import DSIN as TorchDSIN
        from model_evo_harness.models.tensorflow.sequence import DSIN as TensorFlowDSIN

        target = torch.randn(2, 4)
        sessions = torch.randn(2, 3, 2, 4)
        mask = torch.ones(2, 3, 2, dtype=torch.bool)
        model = TorchDSIN(4, 8, 2)
        model(target, sessions, mask).sum().backward()
        reverse = model.evolution.weight_ih_l0_reverse.grad
        self.assertIsNotNone(reverse)
        self.assertGreater(float(reverse.abs().sum()), 0)

        tf_model = TensorFlowDSIN(4, 8, 2)
        with tf.GradientTape() as tape:
            output = tf_model(tf.constant(target.numpy()), tf.constant(sessions.numpy()),
                              tf.constant(mask.numpy()))
            objective = tf.reduce_sum(output)
        backward_variables = tf_model.evolution.backward_layer.trainable_variables
        gradients = tape.gradient(objective, backward_variables)
        self.assertTrue(any(g is not None and float(tf.reduce_sum(tf.abs(g))) > 0
                            for g in gradients))

    def test_tensorflow_dien_dynamic_sequence_length(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.sequence import DIEN

        model = DIEN(4, 8)

        @tf.function(input_signature=[
            tf.TensorSpec((None, 4), tf.float32),
            tf.TensorSpec((None, None, 4), tf.float32),
            tf.TensorSpec((None, None), tf.bool),
        ])
        def run(target, history, mask):
            return model(target, history, mask)

        for length in (3, 5):
            output = run(tf.random.normal((2, 4)), tf.random.normal((2, length, 4)),
                         tf.ones((2, length), dtype=tf.bool))
            self.assertEqual(tuple(output.shape), (2,))
            self.assertTrue(np.isfinite(output.numpy()).all())

    def test_causal_encoders_do_not_use_future_events_for_earlier_positions(self):
        import torch
        from model_evo_harness.models.pytorch.sequence import HSTU, SASRec

        history = torch.randn(1, 4, 4)
        changed = history.clone()
        changed[:, 3] += 100
        mask = torch.ones((1, 4), dtype=torch.bool)
        for model, encode in (
                (SASRec(4, 2, 4), lambda m, h: m.encode(h, mask)),
                (HSTU(4), lambda m, h: m(h, torch.arange(4)[None, :], mask))):
            with self.subTest(model=type(model).__name__):
                first = encode(model, history)[:, :3]
                second = encode(model, changed)[:, :3]
                self.assertTrue(torch.allclose(first, second, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
