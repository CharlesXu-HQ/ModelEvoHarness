"""Smoke tests for framework-native reference models."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class PyTorchModelsTests(unittest.TestCase):
    def test_models_shapes_and_gradients(self):
        import torch
        from model_evo_harness.models.pytorch import DCN, DIN, FM, MMoE, DeepFM, TwoTower

        torch.manual_seed(4)
        fields = torch.randint(0, 4, (5, 3))
        dense = torch.randn(5, 8)
        target = torch.randn(5, 8)
        history = torch.randn(5, 4, 8)
        mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1],
                             [1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        models_and_outputs = [
            (FM([4, 4, 4], 4), lambda m: m(fields), (5,)),
            (DeepFM([4, 4, 4], 4, 8), lambda m: m(fields), (5,)),
            (DCN(8, 2, 12), lambda m: m(dense), (5,)),
            (DIN(8, 12), lambda m: m(target, history, mask), (5,)),
            (MMoE(8, 2, 3, 12), lambda m: m(dense), (5, 2)),
            (TwoTower(8, 8, 6, 12), lambda m: m(dense, dense), (5, 5)),
        ]
        for model, call, shape in models_and_outputs:
            with self.subTest(model=type(model).__name__):
                output = call(model)
                self.assertEqual(tuple(output.shape), shape)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
        din = DIN(8, 12)
        zero_mask = torch.zeros(1, 4, dtype=torch.bool)
        self.assertTrue(torch.allclose(din(target[:1], history[:1], zero_mask),
                                       din(target[:1], 100 * history[:1], zero_mask)))
        self.assertEqual(tuple(TwoTower(8, 8, 6, 12)(dense[:3], dense).shape),
                         (3, 5))


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
class TensorFlowModelsTests(unittest.TestCase):
    def test_models_shapes_and_gradients(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow import DCN, DIN, FM, MMoE, DeepFM, TwoTower

        tf.random.set_seed(4)
        fields = tf.constant([[0, 1, 2]] * 5, dtype=tf.int32)
        dense = tf.random.normal((5, 8))
        target = tf.random.normal((5, 8))
        history = tf.random.normal((5, 4, 8))
        mask = tf.cast([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1],
                        [1, 0, 0, 0], [1, 1, 1, 0]], tf.bool)
        models_and_outputs = [
            (FM([4, 4, 4], 4), lambda m: m(fields), (5,)),
            (DeepFM([4, 4, 4], 4, 8), lambda m: m(fields), (5,)),
            (DCN(8, 2, 12), lambda m: m(dense), (5,)),
            (DIN(8, 12), lambda m: m(target, history, mask), (5,)),
            (MMoE(8, 2, 3, 12), lambda m: m(dense), (5, 2)),
            (TwoTower(8, 8, 6, 12), lambda m: m((dense, dense)), (5, 5)),
        ]
        for model, call, shape in models_and_outputs:
            with self.subTest(model=type(model).__name__):
                with tf.GradientTape() as tape:
                    output = call(model)
                    loss = tf.reduce_mean(tf.square(output))
                self.assertEqual(tuple(output.shape), shape)
                self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(output))))
                gradients = tape.gradient(loss, model.trainable_variables)
                self.assertTrue(any(g is not None and bool(tf.reduce_all(tf.math.is_finite(g)))
                                    for g in gradients))
        din = DIN(8, 12)
        zero_mask = tf.zeros((1, 4), dtype=tf.bool)
        self.assertTrue(bool(tf.reduce_all(tf.equal(
            din(target[:1], history[:1], zero_mask),
            din(target[:1], 100 * history[:1], zero_mask)))))
        self.assertEqual(tuple(TwoTower(8, 8, 6, 12)((dense[:3], dense)).shape),
                         (3, 5))


if __name__ == "__main__":
    unittest.main()
