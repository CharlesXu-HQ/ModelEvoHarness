"""Framework-native references for multi-task, scenario and slate methods."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class PyTorchMultitaskTests(unittest.TestCase):
    def test_star_uses_domain_moments_and_running_stats(self):
        import torch
        from model_evo_harness.models.pytorch.multitask import STAR

        model = STAR(2, 3, 4)
        x = torch.tensor([[1., 2.], [3., 4.], [101., 202.], [103., 204.]])
        domain = torch.tensor([0, 0, 1, 1])
        normalized = model.norm(x, domain)
        self.assertTrue(torch.allclose(normalized[:2].mean(0), torch.zeros(2), atol=1e-5))
        self.assertTrue(torch.allclose(normalized[2:].mean(0), torch.zeros(2), atol=1e-5))
        self.assertTrue(torch.allclose(normalized[:2].var(0, unbiased=False),
                                        torch.ones(2), atol=1e-4))
        self.assertTrue(torch.allclose(model.norm.running_mean[0], torch.tensor([2., 3.])))
        self.assertTrue(torch.allclose(model.norm.running_mean[1], torch.tensor([102., 203.])))
        model.eval()
        alone = model.norm(torch.tensor([[5., 7.]]), torch.tensor([0]))
        with_other = model.norm(torch.tensor([[5., 7.], [999., 999.]]),
                                torch.tensor([0, 1]))[:1]
        self.assertTrue(torch.allclose(alone, with_other))
        model.train()
        rare = model.norm(torch.tensor([[7., 11.]]), torch.tensor([2]))
        self.assertTrue(torch.isfinite(rare).all())
        self.assertTrue(torch.allclose(rare, torch.tensor([[7., 11.]]), atol=1e-5))

    def test_aitm_attention_and_order_loss(self):
        import torch
        from model_evo_harness.models.pytorch.multitask import AITM, aitm_loss

        model = AITM(5, 3, 4)
        weights = model.attention_weights(torch.randn(2, 4), torch.randn(2, 4), 0)
        self.assertEqual(tuple(weights.shape), (2, 2))
        self.assertTrue(torch.allclose(weights.sum(1), torch.ones(2)))
        self.assertGreaterEqual(float(weights.detach().min()), 0)
        labels = torch.tensor([[1., 1., 0.]])
        ordered = torch.tensor([[3., 1., -1.]])
        reversed_order = torch.tensor([[-1., 1., 3.]])
        ordered_penalty = aitm_loss(ordered, labels, alpha=1) - aitm_loss(
            ordered, labels, alpha=0)
        reversed_penalty = aitm_loss(reversed_order, labels, alpha=1) - aitm_loss(
            reversed_order, labels, alpha=0)
        self.assertAlmostEqual(float(ordered_penalty), 0, places=6)
        self.assertGreater(float(reversed_penalty), 0)

    def test_prm_rejects_empty_candidate_slate(self):
        import torch
        from model_evo_harness.models.pytorch.multitask import PRM

        model = PRM(8, 4, 12, 5, 2)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 5, 8), torch.randn(2, 4),
                  torch.tensor([[True, False, False, False, False], [False] * 5]))

    def test_shapes_and_gradients(self):
        import torch
        from model_evo_harness.models.pytorch.multitask import (
            AITM, APG, ESMM, HMoE, M2M, PEPNet, PLE, PRM, PRS, STAR, SharedBottom)

        torch.manual_seed(3)
        x = torch.randn(4, 8)
        scenario = torch.randn(4, 4)
        domain = torch.tensor([0, 1, 0, 1])
        slate = torch.randn(4, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0]] * 4, dtype=torch.bool)
        cases = [
            (SharedBottom(8, 3, 12), lambda m: m(x), (4, 3)),
            (PLE(8, 3, 2, 1, 6, levels=2), lambda m: m(x), (4, 3)),
            (AITM(8, 3, 12), lambda m: m(x), (4, 3)),
            (M2M(8, 4, 3, 2, 6, 8), lambda m: m(x, scenario), (4, 3)),
            (APG(8, 4, 4, 8), lambda m: m(x, scenario), (4,)),
            (HMoE(8, 3, 2, 6), lambda m: m(x, domain), (4,)),
            (PEPNet(8, 4, 3, 12), lambda m: m(x, scenario, scenario), (4, 3)),
            (STAR(8, 3, 12), lambda m: m(x, domain), (4,)),
            (PRM(8, 4, 12, 5, 2), lambda m: m(slate, scenario, mask), (4, 5)),
            (PRS(8, 12, 5), lambda m: m(slate)["ctr"], (4, 5)),
        ]
        for model, call, shape in cases:
            with self.subTest(model=type(model).__name__):
                output = call(model)
                self.assertEqual(tuple(output.shape), shape)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
        esmm = ESMM(8, 12)
        result = esmm(x)
        self.assertEqual(set(result), {"click", "post_click", "joint"})
        self.assertEqual(tuple(result["joint"].shape), (4,))
        self.assertTrue(torch.allclose(result["joint"],
                                       result["click"] * result["post_click"]))
        result["joint"].sum().backward()
        self.assertTrue(any(p.grad is not None for p in esmm.parameters()))

    def test_core_mechanisms(self):
        import torch
        from model_evo_harness.models.pytorch.multitask import (
            AITM, APG, HMoE, M2M, PEPNet, PLE, PRM, PRS, STAR, SharedBottom)

        torch.manual_seed(7)
        x = torch.randn(4, 8)
        scenario = torch.randn(4, 4)
        apg = APG(8, 4, 4, 8)
        self.assertEqual(tuple(apg.generated_matrix(scenario).shape), (4, 4, 4))
        self.assertFalse(torch.allclose(apg(x, scenario), apg(x, scenario + 2)))

        shared = SharedBottom(8, 2, 12)
        shared(x)[:, 0].sum().backward()
        self.assertGreater(shared.bottom[0].weight.grad.abs().sum().item(), 0)

        m2m = M2M(8, 4, 2, 3, 6, 8)
        self.assertFalse(torch.allclose(m2m(x, scenario), m2m(x, scenario + 2)))

        pepnet = PEPNet(8, 4, 2, 12)
        base = pepnet(x, scenario, scenario)
        self.assertFalse(torch.allclose(base, pepnet(x, scenario + 2, scenario)))
        self.assertFalse(torch.allclose(base, pepnet(x, scenario, scenario + 2)))

        star = STAR(8, 2, 12)
        self.assertFalse(torch.allclose(
            star(x, torch.zeros(4, dtype=torch.long)),
            star(x, torch.ones(4, dtype=torch.long))))

        aitm = AITM(8, 3, 12)
        aitm(x)[:, 1].sum().backward()
        self.assertGreater(aitm.task_encoders[0][0].weight.grad.abs().sum().item(), 0)

        hmoe = HMoE(8, 2, 2, 6)
        hmoe(x, torch.zeros(4, dtype=torch.long)).sum().backward()
        own = hmoe.domain_towers[0].weight.grad
        cross = hmoe.domain_towers[1].weight.grad
        self.assertGreater(own.abs().sum().item(), 0)
        self.assertTrue(cross is None or torch.count_nonzero(cross) == 0)

        ple = PLE(8, 2, 2, 1, 6, levels=1)
        ple(x)[:, 0].sum().backward()
        other_grad = ple.task_experts[0][1][0][0].weight.grad
        self.assertTrue(other_grad is None or torch.count_nonzero(other_grad) == 0)

        mask = torch.tensor([[1, 1, 0, 0, 0]] * 4, dtype=torch.bool)
        prm = PRM(8, 4, 12, 5, 2)
        probabilities = prm(torch.randn(4, 5, 8), scenario, mask)
        self.assertTrue(torch.allclose(probabilities[:, 2:], torch.zeros(4, 3)))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(4)))
        probabilities[:, 0].sum().backward()
        self.assertGreater(prm.attention.in_proj_weight.grad.abs().sum().item(), 0)

        prs = PRS(8, 12, 5)
        ctr = torch.tensor([[0.4, 0.5]])
        continuation = torch.tensor([[0.2, 0.9]])
        self.assertTrue(torch.allclose(prs.slate_reward(ctr, continuation),
                                       torch.tensor([0.5])))
        order = prs.rerank(torch.randn(2, 4, 8), beam_size=2)
        self.assertEqual(tuple(order.shape), (2, 4))
        self.assertEqual([sorted(row) for row in order.tolist()], [list(range(4))] * 2)
        probabilities = prs(torch.randn(4, 5, 8))
        prs.slate_reward(probabilities["ctr"], probabilities["continuation"]).sum().backward()
        self.assertGreater(prs.continuation_head.weight.grad.abs().sum().item(), 0)


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
class TensorFlowMultitaskTests(unittest.TestCase):
    def test_star_uses_domain_moments_and_running_stats(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.multitask import STAR

        model = STAR(2, 3, 4)
        x = tf.constant([[1., 2.], [3., 4.], [101., 202.], [103., 204.]])
        domain = tf.constant([0, 0, 1, 1])
        normalized = model.norm(x, domain, training=True)
        self.assertTrue(bool(tf.reduce_all(tf.abs(
            tf.reduce_mean(normalized[:2], axis=0)) < 1e-5)))
        self.assertTrue(bool(tf.reduce_all(tf.abs(
            tf.reduce_mean(normalized[2:], axis=0)) < 1e-5)))
        self.assertTrue(bool(tf.reduce_all(tf.abs(model.norm.running_mean[0]
                                                - tf.constant([2., 3.])) < 1e-5)))
        alone = model.norm(tf.constant([[5., 7.]]), tf.constant([0]), training=False)
        with_other = model.norm(tf.constant([[5., 7.], [999., 999.]]),
                                tf.constant([0, 1]), training=False)[:1]
        self.assertTrue(bool(tf.reduce_all(tf.abs(alone - with_other) < 1e-5)))
        rare = model.norm(tf.constant([[7., 11.]]), tf.constant([2]), training=True)
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(rare))))
        self.assertTrue(bool(tf.reduce_all(tf.abs(rare - [[7., 11.]]) < 1e-5)))

    def test_star_partitioned_norm_runs_in_graph_mode(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.multitask import STAR

        model = STAR(2, 2, 4)

        @tf.function
        def train(x, domain):
            return model(x, domain, training=True)

        @tf.function
        def evaluate(x, domain):
            return model(x, domain, training=False)

        x = tf.constant([[1., 2.], [3., 4.], [101., 202.], [103., 204.]])
        domain = tf.constant([0, 0, 1, 1])
        self.assertEqual(tuple(train(x, domain).shape), (4,))
        self.assertEqual(tuple(evaluate(x, domain).shape), (4,))
        self.assertTrue(bool(tf.reduce_all(tf.abs(model.norm.running_mean[0]
                                                - tf.constant([2., 3.])) < 1e-5)))

    def test_aitm_attention_and_order_loss(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.multitask import AITM, aitm_loss

        model = AITM(5, 3, 4)
        weights = model.attention_weights(tf.random.normal((2, 4)),
                                          tf.random.normal((2, 4)), 0)
        self.assertEqual(tuple(weights.shape), (2, 2))
        self.assertTrue(bool(tf.reduce_all(tf.abs(tf.reduce_sum(weights, axis=1) - 1)
                                           < 1e-5)))
        labels = tf.constant([[1., 1., 0.]])
        ordered = tf.constant([[3., 1., -1.]])
        reversed_order = tf.constant([[-1., 1., 3.]])
        ordered_penalty = aitm_loss(ordered, labels, alpha=1) - aitm_loss(
            ordered, labels, alpha=0)
        reversed_penalty = aitm_loss(reversed_order, labels, alpha=1) - aitm_loss(
            reversed_order, labels, alpha=0)
        self.assertAlmostEqual(float(ordered_penalty), 0, places=6)
        self.assertGreater(float(reversed_penalty), 0)

    def test_prm_rejects_empty_candidate_slate(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.multitask import PRM

        model = PRM(8, 4, 12, 5, 2)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model(tf.random.normal((2, 5, 8)), tf.random.normal((2, 4)),
                  tf.constant([[True, False, False, False, False], [False] * 5]))

    def test_shapes_gradients_and_mechanisms(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.multitask import (
            AITM, APG, ESMM, HMoE, M2M, PEPNet, PLE, PRM, PRS, STAR, SharedBottom)

        tf.random.set_seed(3)
        x = tf.random.normal((4, 8))
        scenario = tf.random.normal((4, 4))
        domain = tf.constant([0, 1, 0, 1])
        slate = tf.random.normal((4, 5, 8))
        mask = tf.constant([[True, True, True, False, False]] * 4)
        cases = [
            (SharedBottom(8, 3, 12), lambda m: m(x), (4, 3)),
            (PLE(8, 3, 2, 1, 6, levels=2), lambda m: m(x), (4, 3)),
            (AITM(8, 3, 12), lambda m: m(x), (4, 3)),
            (M2M(8, 4, 3, 2, 6, 8), lambda m: m(x, scenario), (4, 3)),
            (APG(8, 4, 4, 8), lambda m: m(x, scenario), (4,)),
            (HMoE(8, 3, 2, 6), lambda m: m(x, domain), (4,)),
            (PEPNet(8, 4, 3, 12), lambda m: m(x, scenario, scenario), (4, 3)),
            (STAR(8, 3, 12), lambda m: m(x, domain), (4,)),
            (PRM(8, 4, 12, 5, 2), lambda m: m(slate, scenario, mask), (4, 5)),
            (PRS(8, 12, 5), lambda m: m(slate)["ctr"], (4, 5)),
        ]
        for model, call, shape in cases:
            with self.subTest(model=type(model).__name__):
                with tf.GradientTape() as tape:
                    output = call(model)
                    loss = tf.reduce_mean(tf.square(output))
                self.assertEqual(tuple(output.shape), shape)
                self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(output))))
                gradients = tape.gradient(loss, model.trainable_variables)
                self.assertTrue(any(g is not None and bool(tf.reduce_all(tf.math.is_finite(g)))
                                    for g in gradients))
        esmm = ESMM(8, 12)
        with tf.GradientTape() as tape:
            result = esmm(x)
            joint_loss = tf.reduce_sum(result["joint"])
        self.assertTrue(bool(tf.reduce_all(tf.equal(
            result["joint"], result["click"] * result["post_click"]))))
        esmm_grads = tape.gradient(joint_loss, esmm.trainable_variables)
        self.assertTrue(all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g)))
                            for g in esmm_grads))
        apg = APG(8, 4, 4, 8)
        self.assertEqual(tuple(apg.generated_matrix(scenario).shape), (4, 4, 4))
        self.assertFalse(bool(tf.reduce_all(tf.equal(apg(x, scenario), apg(x, scenario + 2)))))
        shared = SharedBottom(8, 2, 12)
        with tf.GradientTape() as tape:
            first_task = tf.reduce_sum(shared(x)[:, 0])
        self.assertTrue(any(g is not None and bool(tf.reduce_any(tf.not_equal(g, 0)))
                            for g in tape.gradient(first_task, shared.bottom.trainable_variables)))
        aitm = AITM(8, 3, 12)
        with tf.GradientTape() as tape:
            later_task = tf.reduce_sum(aitm(x)[:, 1])
        self.assertTrue(any(g is not None and bool(tf.reduce_any(tf.not_equal(g, 0)))
                            for g in tape.gradient(later_task,
                                                   aitm.task_encoders[0].trainable_variables)))
        ple = PLE(8, 2, 2, 1, 6, levels=1)
        with tf.GradientTape() as tape:
            first_task = tf.reduce_sum(ple(x)[:, 0])
        unrelated = tape.gradient(first_task, ple.task_experts[0][1][0].trainable_variables)
        self.assertTrue(all(g is None or bool(tf.reduce_all(tf.equal(g, 0)))
                            for g in unrelated))
        hmoe = HMoE(8, 2, 2, 6)
        with tf.GradientTape() as tape:
            own_domain = tf.reduce_sum(hmoe(x, tf.zeros(4, dtype=tf.int32)))
        cross = tape.gradient(own_domain, hmoe.domain_towers[1].trainable_variables)
        self.assertTrue(all(g is None or bool(tf.reduce_all(tf.equal(g, 0))) for g in cross))
        m2m = M2M(8, 4, 2, 3, 6, 8)
        self.assertFalse(bool(tf.reduce_all(tf.equal(m2m(x, scenario),
                                                 m2m(x, scenario + 2)))))
        pepnet = PEPNet(8, 4, 2, 12)
        base = pepnet(x, scenario, scenario)
        self.assertFalse(bool(tf.reduce_all(tf.equal(base, pepnet(x, scenario + 2, scenario)))))
        self.assertFalse(bool(tf.reduce_all(tf.equal(base, pepnet(x, scenario, scenario + 2)))))
        star = STAR(8, 2, 12)
        self.assertFalse(bool(tf.reduce_all(tf.equal(
            star(x, tf.zeros(4, dtype=tf.int32)),
            star(x, tf.ones(4, dtype=tf.int32))))))
        prm = PRM(8, 4, 12, 5, 2)
        p = prm(slate, scenario, mask)
        self.assertTrue(bool(tf.reduce_all(tf.equal(p[:, 3:], 0))))
        self.assertTrue(bool(tf.reduce_all(tf.abs(tf.reduce_sum(p, axis=1) - 1) < 1e-6)))
        with tf.GradientTape() as tape:
            attention_output = tf.reduce_sum(prm(slate, scenario, mask)[:, 0])
        self.assertTrue(any(g is not None and bool(tf.reduce_any(tf.not_equal(g, 0)))
                            for g in tape.gradient(attention_output,
                                                   prm.attention.trainable_variables)))
        prs = PRS(8, 12, 5)
        self.assertTrue(bool(tf.reduce_all(tf.abs(prs.slate_reward(
            tf.constant([[0.4, 0.5]]), tf.constant([[0.2, 0.9]])) - 0.5) < 1e-6)))
        order = prs.rerank(tf.random.normal((2, 4, 8)), beam_size=2)
        self.assertEqual(tuple(order.shape), (2, 4))
        self.assertEqual([sorted(row) for row in order.numpy().tolist()],
                         [list(range(4))] * 2)
        with tf.GradientTape() as tape:
            probabilities = prs(slate)
            reward = tf.reduce_sum(prs.slate_reward(probabilities["ctr"],
                                                    probabilities["continuation"]))
        self.assertTrue(any(g is not None and bool(tf.reduce_any(tf.not_equal(g, 0)))
                            for g in tape.gradient(reward,
                                                   prs.continuation_head.trainable_variables)))


if __name__ == "__main__":
    unittest.main()
