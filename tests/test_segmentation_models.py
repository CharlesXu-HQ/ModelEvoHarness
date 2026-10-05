import unittest

import numpy as np


class MLRTests(unittest.TestCase):
    def test_pytorch_regions_are_normalized_and_trainable(self):
        import torch
        from model_evo_harness.models.pytorch.segmentation import MLR

        model = MLR(input_dim=3, num_regions=4)
        x = torch.randn(6, 3, requires_grad=True)
        weights = model.region_weights(x)
        self.assertEqual(tuple(weights.shape), (6, 4))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(6)))
        prediction = model(x)
        self.assertEqual(tuple(prediction.shape), (6,))
        prediction.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_tensorflow_regions_are_normalized_and_trainable(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.segmentation import MLR

        model = MLR(input_dim=3, num_regions=4)
        x = tf.Variable(np.random.randn(6, 3).astype("float32"))
        with tf.GradientTape() as tape:
            weights = model.region_weights(x)
            prediction = model(x)
            total = tf.reduce_sum(prediction)
        self.assertEqual(tuple(weights.shape), (6, 4))
        self.assertTrue(np.allclose(tf.reduce_sum(weights, axis=-1).numpy(), 1))
        self.assertEqual(tuple(prediction.shape), (6,))
        gradients = tape.gradient(total, [x] + model.trainable_variables)
        self.assertTrue(all(g is not None and np.isfinite(g.numpy()).all() for g in gradients))


if __name__ == "__main__":
    unittest.main()
