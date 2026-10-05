import unittest

import numpy as np


class TrainingMethodTests(unittest.TestCase):
    def test_pytorch_losses_have_expected_controls_and_gradients(self):
        import torch
        from torch.nn import functional as F
        from model_evo_harness.models.pytorch.training import (
            bpr_loss, focal_loss, listwise_loss)

        logits = torch.tensor([0.4, -0.2], requires_grad=True)
        labels = torch.tensor([1.0, 0.0])
        self.assertTrue(torch.allclose(focal_loss(logits, labels, gamma=0, alpha=0.5),
                                       0.5 * F.binary_cross_entropy_with_logits(logits, labels)))
        self.assertLess(bpr_loss(torch.tensor([3.0]), torch.tensor([1.0])).item(),
                        bpr_loss(torch.tensor([1.0]), torch.tensor([3.0])).item())
        scores = torch.tensor([[3.0, 1.0, 100.0]], requires_grad=True)
        relevance = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        self.assertLess(listwise_loss(scores, relevance, mask).item(),
                        listwise_loss(-scores, relevance, mask).item())
        (focal_loss(logits, labels) + listwise_loss(scores, relevance, mask)).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_tensorflow_losses_have_expected_controls_and_gradients(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.training import (
            bpr_loss, focal_loss, listwise_loss)

        logits = tf.Variable([0.4, -0.2])
        labels = tf.constant([1.0, 0.0])
        with tf.GradientTape() as tape:
            focal = focal_loss(logits, labels, gamma=0, alpha=0.5)
            scores = tf.constant([[3.0, 1.0, 100.0]])
            ranking = listwise_loss(scores, tf.constant([[1.0, 0.0, 0.0]]),
                                    tf.constant([[True, True, False]]))
            total = focal + ranking
        expected = 0.5 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits))
        self.assertAlmostEqual(float(focal), float(expected), places=6)
        self.assertLess(float(bpr_loss(tf.constant([3.0]), tf.constant([1.0]))),
                        float(bpr_loss(tf.constant([1.0]), tf.constant([3.0]))))
        self.assertLess(float(ranking), float(listwise_loss(-scores,
                        tf.constant([[1.0, 0.0, 0.0]]),
                        tf.constant([[True, True, False]]))))
        self.assertTrue(np.isfinite(tape.gradient(total, logits).numpy()).all())

    def test_hard_mining_uses_only_host_allowed_negative_candidates(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.training import hard_negative_indices as torch_mine
        from model_evo_harness.models.tensorflow.training import hard_negative_indices as tf_mine

        query = [[1.0, 0.0]]
        items = [[0.9, 0.0], [0.8, 0.0], [0.7, 0.0], [0.6, 0.0]]
        allowed = [[False, True, False, True]]  # 0 is positive, 2 unexposed.
        self.assertEqual(torch_mine(torch.tensor(query), torch.tensor(items),
                                    torch.tensor(allowed), k=2).tolist(), [[1, 3]])
        self.assertEqual(tf_mine(tf.constant(query), tf.constant(items),
                                 tf.constant(allowed), k=2).numpy().tolist(), [[1, 3]])

    def test_ineligible_negatives_and_empty_positive_slates_are_rejected(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch import training as pt
        from model_evo_harness.models.tensorflow import training as kt

        queries = [[1.0, 0.0]]
        items = [[1.0, 0.0], [0.5, 0.0], [0.1, 0.0]]
        allowed = [[True, False, False]]
        with self.assertRaises(ValueError):
            pt.hard_negative_indices(torch.tensor(queries), torch.tensor(items),
                                     torch.tensor(allowed), k=2)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            kt.hard_negative_indices(tf.constant(queries), tf.constant(items),
                                     tf.constant(allowed), k=2)
        with self.assertRaises(ValueError):
            pt.listwise_loss(torch.tensor([[1.0, 2.0]]), torch.zeros(1, 2),
                             torch.tensor([[True, True]]))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            kt.listwise_loss(tf.constant([[1.0, 2.0]]), tf.zeros((1, 2)),
                             tf.constant([[True, True]]))


if __name__ == "__main__":
    unittest.main()
