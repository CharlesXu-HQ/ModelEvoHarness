import unittest

import numpy as np


class PaperRankingTests(unittest.TestCase):
    def test_torch_forward_gradients_and_fm_loading(self):
        import torch
        from model_evo_harness.models.pytorch.paper_ranking import (
            DeepCrossing, FNN, LatentCross, MultiViewDSSM, NeuMF, PredictiveBilinear)

        x = torch.randn(3, 4, requires_grad=True)
        models = [DeepCrossing(4, 8), PredictiveBilinear(4, 4)]
        for model in models:
            output = model(x) if isinstance(model, DeepCrossing) else model(x, x)
            self.assertEqual(tuple(output.shape), (3,))
            output.sum().backward(retain_graph=True)
        self.assertGreater(float(x.grad.abs().sum()), 0)
        fnn = FNN((3, 4), 2)
        weights = [torch.ones(3, 2), torch.full((4, 2), 2.)]
        fnn.load_fm_embeddings(weights)
        self.assertTrue(torch.equal(fnn.embeddings[0].weight, weights[0]))
        fnn(torch.tensor([[0, 1], [1, 2]])).sum().backward()
        self.assertIsNotNone(fnn.embeddings[0].weight.grad)
        self.assertEqual(tuple(NeuMF(3, 4, 2)(torch.tensor([0, 1]),
                                                  torch.tensor([2, 3])).shape), (2,))
        latent = LatentCross(4, 2)
        self.assertEqual(tuple(latent(x[:, None], torch.ones(3, 2)).shape), (3, 1, 4))
        tower = MultiViewDSSM(4, (3, 5), 4)
        self.assertEqual(tuple(tower(x, torch.ones(3, 5), 1).shape), (3,))
        self.assertTrue(torch.allclose(tower.encode_user(x).norm(dim=-1),
                                       torch.ones(3), atol=1e-5))

    def test_tensorflow_forward_gradients_and_fm_loading(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.paper_ranking import (
            DeepCrossing, FNN, LatentCross, MultiViewDSSM, NeuMF, PredictiveBilinear)

        x = tf.Variable(tf.random.normal((3, 4)))
        with tf.GradientTape() as tape:
            a = DeepCrossing(4, 8)(x)
            b = PredictiveBilinear(4, 4)(x, x)
            loss = tf.reduce_sum(a + b)
        self.assertGreater(float(tf.reduce_sum(tf.abs(tape.gradient(loss, x)))), 0)
        fnn = FNN((3, 4), 2)
        fnn.load_fm_embeddings([tf.ones((3, 2)), tf.fill((4, 2), 2.)])
        self.assertTrue(np.allclose(fnn.embeddings[0].embeddings.numpy(), 1))
        self.assertEqual(tuple(fnn(tf.constant([[0, 1], [1, 2]])).shape), (2,))
        self.assertEqual(tuple(NeuMF(3, 4, 2)(tf.constant([0, 1]),
                                                  tf.constant([2, 3])).shape), (2,))
        self.assertEqual(tuple(LatentCross(4, 2)(x[:, None], tf.ones((3, 2))).shape),
                         (3, 1, 4))
        tower = MultiViewDSSM(4, (3, 5), 4)
        self.assertEqual(tuple(tower(x, tf.ones((3, 5)), domain=1).shape), (3,))
        self.assertTrue(np.allclose(tf.norm(tower.encode_user(x), axis=-1), 1, atol=1e-5))

    def test_dpp_mask_singular_and_ranking_losses(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.paper_ranking import (
            DPPGreedy as PTDPP, RankNet as PTRank, LambdaRank as PTLambda)
        from model_evo_harness.models.tensorflow.paper_ranking import (
            DPPGreedy as TFDPP, RankNet as TFRank, LambdaRank as TFLambda)

        kernel = np.array([[[1., 1., 0.], [1., 1., 0.], [0., 0., 2.]]], np.float32)
        eligible = np.array([[True, True, False]])
        expected = [[0, -1, -1]]
        self.assertEqual(PTDPP(3)(torch.tensor(kernel), torch.tensor(eligible)).tolist(), expected)
        self.assertEqual(TFDPP(3)(tf.constant(kernel), tf.constant(eligible)).numpy().tolist(), expected)
        scores = np.array([[2., 0., 100.], [3., 2., 1.]], np.float32)
        labels = np.array([[2., 0., 9.], [1., 1., 0.]], np.float32)
        mask = np.array([[True, True, False], [True, True, True]])
        ts = torch.tensor(scores, requires_grad=True)
        for model in (PTRank(), PTLambda()):
            value = model(ts, torch.tensor(labels), torch.tensor(mask))
            self.assertTrue(torch.isfinite(value))
            value.backward(retain_graph=True)
        self.assertEqual(float(ts.grad[0, 2]), 0.)
        for model in (TFRank(), TFLambda()):
            variable = tf.Variable(scores)
            with tf.GradientTape() as tape:
                value = model(variable, tf.constant(labels), tf.constant(mask))
            grad = tape.gradient(value, variable)
            self.assertTrue(np.isfinite(float(value)))
            self.assertEqual(float(grad[0, 2]), 0.)
        tied = torch.tensor([[1., 1.]])
        self.assertEqual(float(PTRank()(tied, tied, torch.ones(1, 2, dtype=torch.bool))), 0.)

    def test_numeric_ranknet_and_bilinear(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.paper_ranking import (
            RankNet as PTRank, PredictiveBilinear as PTBilinear)
        from model_evo_harness.models.tensorflow.paper_ranking import (
            RankNet as TFRank, PredictiveBilinear as TFBilinear)

        scores = np.array([[1., 0.]], np.float32)
        labels = np.array([[1., 0.]], np.float32)
        mask = np.array([[True, True]])
        expected = np.logaddexp(0., -1.)
        self.assertAlmostEqual(float(PTRank()(torch.tensor(scores), torch.tensor(labels),
                                              torch.tensor(mask))), expected, places=6)
        self.assertAlmostEqual(float(TFRank()(tf.constant(scores), tf.constant(labels),
                                              tf.constant(mask))), expected, places=6)
        user = np.array([[2., 3.]], np.float32)
        item = np.array([[4., 5.]], np.float32)
        for model, tensor in ((PTBilinear(2, 2), torch.tensor),
                              (TFBilinear(2, 2), tf.constant)):
            model.weight.assign(tf.eye(2)) if hasattr(model.weight, "assign") else None
            if hasattr(model.weight, "data"):
                model.weight.data.copy_(torch.eye(2))
            value = model(tensor(user), tensor(item))[0]
            if hasattr(value, "detach"):
                value = value.detach()
            self.assertAlmostEqual(float(value), 23., places=5)

    def test_lambda_preserves_swap_ndcg_magnitude(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.paper_ranking import LambdaRank as PT
        from model_evo_harness.models.tensorflow.paper_ranking import LambdaRank as TF

        scores = np.array([[1., 0.]], np.float32)
        labels = np.array([[1., 0.]], np.float32)
        mask = np.array([[True, True]])
        delta = 1 - 1 / np.log2(3)
        expected = np.logaddexp(0., -1.) * delta
        self.assertAlmostEqual(float(PT()(torch.tensor(scores), torch.tensor(labels),
                                          torch.tensor(mask))), expected, places=6)
        self.assertAlmostEqual(float(TF()(tf.constant(scores), tf.constant(labels),
                                          tf.constant(mask))), expected, places=6)

    def test_incremental_dpp_near_singular_and_tf_dynamic_graph(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.paper_ranking import DPPGreedy as PT
        from model_evo_harness.models.tensorflow.paper_ranking import DPPGreedy as TF

        kernel = np.array([[[1., 0.999999, 0.], [0.999999, 1., 0.],
                            [0., 0., 0.5]]], np.float32)
        allowed = np.array([[True, True, True]])
        expected = [[0, 2, -1]]
        self.assertEqual(PT(3, tolerance=1e-5)(torch.tensor(kernel),
                         torch.tensor(allowed)).tolist(), expected)
        tf_model = TF(3, tolerance=1e-5)
        @tf.function(input_signature=(tf.TensorSpec((None, None, None), tf.float32),
                                      tf.TensorSpec((None, None), tf.bool)))
        def select(k, m):
            return tf_model(k, m)
        self.assertEqual(select(tf.constant(kernel), tf.constant(allowed)).numpy().tolist(),
                         expected)
        single = tf.eye(2, batch_shape=[2])
        self.assertEqual(tuple(select(single, tf.ones((2, 2), tf.bool)).shape), (2, 3))

    def test_latent_cross_identity_at_zero_context(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.paper_ranking import LatentCross as PT
        from model_evo_harness.models.tensorflow.paper_ranking import LatentCross as TF

        hidden = np.array([[[2., -3.]]], np.float32)
        context = np.array([[0., 0.]], np.float32)
        pt = PT(2, 2)
        with torch.no_grad():
            pt.context.weight.zero_()
            pt.context.bias.zero_()
        self.assertTrue(np.allclose(pt(torch.tensor(hidden), torch.tensor(context)).detach(), hidden))
        model = TF(2, 2)
        model(tf.constant(hidden), tf.constant(context))
        model.context.kernel.assign(tf.zeros_like(model.context.kernel))
        model.context.bias.assign(tf.zeros_like(model.context.bias))
        @tf.function(input_signature=(tf.TensorSpec((None, None, 2), tf.float32),
                                      tf.TensorSpec((None, 2), tf.float32)))
        def crossed(h, c):
            return model(h, c)
        self.assertTrue(np.allclose(crossed(tf.constant(hidden), tf.constant(context)), hidden))
