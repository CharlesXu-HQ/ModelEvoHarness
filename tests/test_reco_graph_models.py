"""Numeric and gradient contracts for standalone graph recommendation cores."""

import ast
import importlib.util
from pathlib import Path
import unittest


_ROOT = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models"
_NAMES = {"GraphSAGE", "PinSage", "SDNE", "LINE", "RippleNet", "KGAT", "CDL"}


class GraphSourceTests(unittest.TestCase):
    def test_each_framework_has_standalone_seven_class_reference(self):
        for framework, package in (("pytorch", "torch"), ("tensorflow", "tensorflow")):
            with self.subTest(framework=framework):
                path = _ROOT / framework / "graph.py"
                tree = ast.parse(path.read_text())
                self.assertEqual({n.name for n in tree.body if isinstance(n, ast.ClassDef)}, _NAMES)
                imports = {alias.name.split(".")[0] for node in ast.walk(tree)
                           if isinstance(node, ast.Import) for alias in node.names}
                imports.update(node.module.split(".")[0] for node in ast.walk(tree)
                               if isinstance(node, ast.ImportFrom) and node.module)
                self.assertEqual(imports, {package})


class PyTorchGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("PyTorch is optional")

    def test_aggregation_padding_and_gradients(self):
        import torch
        from model_evo_harness.models.pytorch.graph import GraphSAGE, PinSage

        center = torch.tensor([[1., 2.]], requires_grad=True)
        neighbors = torch.tensor([[[3., 4.], [100., 200.]]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        sage = GraphSAGE(2, 2)
        with torch.no_grad():
            sage.projection.weight.copy_(torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]]))
            sage.projection.bias.zero_()
        output = sage(center, neighbors, mask)
        torch.testing.assert_close(output, torch.tensor([[4., 6.]]) / (52.**0.5))
        output.sum().backward()
        self.assertEqual(neighbors.grad[0, 1].abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(sage.projection.weight.grad).all().item())

        pinsage = PinSage(2, 2, 2)
        with torch.no_grad():
            pinsage.neighbor_projection.weight.copy_(torch.eye(2))
            pinsage.neighbor_projection.bias.zero_()
            pinsage.output_projection.weight.copy_(sage.projection.weight)
            pinsage.output_projection.bias.zero_()
        first = pinsage(center.detach(), neighbors.detach(), torch.tensor([[2., 999.]]), mask)
        second = pinsage(center.detach(), torch.tensor([[[3., 4.], [-99., -99.]]]),
                         torch.tensor([[2., 1.]]), mask)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first, output.detach())
        first.sum().backward()
        self.assertGreater(pinsage.neighbor_projection.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(sage(center.detach(), neighbors.detach(),
                                             torch.zeros_like(mask))).all().item())
        self.assertTrue(torch.isfinite(pinsage(center.detach(), neighbors.detach(),
                                                torch.zeros((1, 2)), torch.zeros_like(mask))).all().item())

    def test_graph_losses_and_kg_attention_are_finite(self):
        import torch
        from model_evo_harness.models.pytorch.graph import CDL, KGAT, LINE, RippleNet, SDNE

        line = LINE(3, 2, order=2)
        with torch.no_grad():
            line.source_embedding.weight.fill_(1.)
            line.context_embedding.weight.fill_(1.)
        line_loss = line.training_loss(torch.tensor([0]), torch.tensor([1]), torch.tensor([[2]]),
                              torch.tensor([2.]))
        expected = torch.nn.functional.softplus(torch.tensor(-2.)) + torch.nn.functional.softplus(torch.tensor(2.))
        torch.testing.assert_close(line_loss, expected)
        line_loss.backward()
        self.assertTrue(torch.isfinite(line.source_embedding.weight.grad).all().item())

        sdne = SDNE(2, 3, 2, beta=3., alpha=2.)
        adjacency = torch.tensor([[0., 1.], [1., 0.]])
        edges = torch.tensor([[0, 1]])
        sdne_loss = sdne.training_loss(adjacency, edges, torch.tensor([1.]))
        self.assertTrue(torch.isfinite(sdne_loss).item())
        sdne_loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in sdne.parameters()))
        exact_sdne = SDNE(2, 3, 2, beta=3., alpha=2.)
        with torch.no_grad():
            for parameter in exact_sdne.parameters():
                parameter.zero_()
        torch.testing.assert_close(exact_sdne.training_loss(adjacency, edges, torch.tensor([1.])),
                                   torch.tensor(5.))

        ripple = RippleNet(4, 1, 2, 2, item_update="plus")
        with torch.no_grad():
            ripple.entity_embedding.weight.copy_(torch.tensor([[1., 0.], [0., 1.],
                                                                [2., 0.], [0., 2.]]))
            ripple.relation_embedding.weight.copy_(torch.eye(2).reshape(1, 4))
        ids = torch.tensor([0])
        heads = torch.tensor([[[0, 1]], [[1, 0]]])
        tails = torch.tensor([[[2, 3]], [[3, 2]]])
        relations = torch.zeros_like(heads)
        mask = torch.tensor([[[True, False]], [[True, False]]])
        score = ripple(ids, heads, relations, tails, mask)
        torch.testing.assert_close(score, torch.sigmoid(torch.tensor([10.])), rtol=1e-5, atol=1e-5)
        score.sum().backward()
        self.assertTrue(torch.isfinite(ripple.entity_embedding.weight.grad).all().item())

        kgat = KGAT(4, 1, 2, 2)
        out = kgat(torch.tensor([0]), torch.tensor([[1, 2]]),
                   torch.tensor([[0, 0]]), torch.tensor([[True, False]]))
        padded = kgat(torch.tensor([0]), torch.tensor([[1, 3]]),
                      torch.tensor([[0, 0]]), torch.tensor([[True, False]]))
        torch.testing.assert_close(out, padded)
        out.sum().backward()
        self.assertTrue(torch.isfinite(kgat.relation_projection.weight.grad).all().item())
        exact_kgat = KGAT(4, 1, 2, 2)
        with torch.no_grad():
            exact_kgat.entity_embedding.weight.copy_(torch.tensor([[1., 0.], [0., 1.],
                                                                    [1., 1.], [0., 0.]]))
            exact_kgat.relation_embedding.weight.zero_()
            exact_kgat.relation_projection.weight.copy_(torch.eye(2).reshape(1, 4))
            exact_kgat.sum_projection.weight.copy_(torch.eye(2))
            exact_kgat.product_projection.weight.copy_(torch.eye(2))
        exact_out = exact_kgat(torch.tensor([0]), torch.tensor([[1, 2]]),
                               torch.tensor([[0, 0]]), torch.tensor([[True, True]]))
        share = torch.sigmoid(torch.tanh(torch.tensor(1.)))
        torch.testing.assert_close(exact_out, torch.stack((1 + 2 * share, torch.tensor(1.)))[None])
        exact_out.sum().backward()
        self.assertGreater(exact_kgat.relation_projection.weight.grad.abs().sum().item(), 0.)

        cdl = CDL(2, 3, 3, 4, 2)
        cdl_loss = cdl.training_loss(torch.tensor([0, 1]), torch.tensor([1, 2]),
                            torch.tensor([[0., 1., 0.], [1., 0., 0.]]),
                            torch.tensor([[0., 1., 1.], [1., 0., 0.]]),
                            torch.tensor([1., 0.]), torch.tensor([2., 0.2]))
        cdl_loss.backward()
        self.assertTrue(torch.isfinite(cdl_loss).item())
        self.assertGreater(cdl.encoder[0].weight.grad.abs().sum().item(), 0.)
        self.assertGreater(cdl.item_embedding.weight.grad.abs().sum().item(), 0.)


class TensorFlowGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("tensorflow") is None:
            raise unittest.SkipTest("TensorFlow is optional")

    def test_aggregation_padding_and_gradients(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.graph import GraphSAGE, PinSage

        center = tf.constant([[1., 2.]])
        neighbors = tf.constant([[[3., 4.], [100., 200.]]])
        mask = tf.constant([[True, False]])
        sage = GraphSAGE(2, 2)
        sage(center, neighbors, mask)
        sage.projection.set_weights([tf.constant([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]).numpy(), tf.zeros((2,)).numpy()])
        with tf.GradientTape() as tape:
            output = sage(center, neighbors, mask)
            loss = tf.reduce_sum(output)
        expected = tf.constant([[4., 6.]]) / tf.sqrt(52.)
        tf.debugging.assert_near(output, expected)
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(tape.gradient(loss, sage.trainable_variables)[0]))))
        pinsage = PinSage(2, 2, 2)
        pinsage(center, neighbors, tf.constant([[2., 999.]]), mask)
        pinsage.neighbor_projection.set_weights([tf.eye(2).numpy(), tf.zeros((2,)).numpy()])
        pinsage.output_projection.set_weights(sage.projection.get_weights())
        first = pinsage(center, neighbors, tf.constant([[2., 999.]]), mask)
        second = pinsage(center, tf.constant([[[3., 4.], [-99., -99.]]]),
                         tf.constant([[2., 1.]]), mask)
        tf.debugging.assert_near(first, second)
        tf.debugging.assert_near(first, output)
        with tf.GradientTape() as tape:
            pin_output = pinsage(center, neighbors, tf.constant([[2., 999.]]), mask)
        pin_gradient = tape.gradient(pin_output, pinsage.neighbor_projection.trainable_variables)[0]
        self.assertTrue(bool(tf.reduce_any(tf.abs(pin_gradient) > 0)))
        tf.debugging.assert_all_finite(sage(center, neighbors, tf.zeros_like(mask)), "GraphSAGE empty")
        tf.debugging.assert_all_finite(pinsage(center, neighbors, tf.zeros((1, 2)),
                                               tf.zeros_like(mask)), "PinSage empty")

    def test_graph_losses_and_kg_attention_are_finite(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.graph import CDL, KGAT, LINE, RippleNet, SDNE

        line = LINE(3, 2, order=2)
        line(tf.constant([0]), tf.constant([1]))
        line.source_embedding.set_weights([tf.ones((3, 2)).numpy()])
        line.context_embedding.set_weights([tf.ones((3, 2)).numpy()])
        with tf.GradientTape() as tape:
            line_loss = line.training_loss(tf.constant([0]), tf.constant([1]), tf.constant([[2]]),
                                  tf.constant([2.]))
        tf.debugging.assert_near(line_loss, tf.nn.softplus(-2.) + tf.nn.softplus(2.))
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g)))
                            for g in tape.gradient(line_loss, line.trainable_variables) if g is not None))

        sdne = SDNE(2, 3, 2, beta=3., alpha=2.)
        adjacency = tf.constant([[0., 1.], [1., 0.]])
        with tf.GradientTape() as tape:
            sdne_loss = sdne.training_loss(adjacency, tf.constant([[0, 1]]), tf.constant([1.]))
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g)))
                            for g in tape.gradient(sdne_loss, sdne.trainable_variables) if g is not None))
        exact_sdne = SDNE(2, 3, 2, beta=3., alpha=2.)
        exact_sdne(adjacency)
        for weight in exact_sdne.trainable_variables:
            weight.assign(tf.zeros_like(weight))
        tf.debugging.assert_near(exact_sdne.training_loss(adjacency, tf.constant([[0, 1]]),
                                                           tf.constant([1.])), 5.)

        ripple = RippleNet(4, 1, 2, 2, item_update="plus")
        ids = tf.constant([0])
        heads = tf.constant([[[0, 1]], [[1, 0]]])
        tails = tf.constant([[[2, 3]], [[3, 2]]])
        relations = tf.zeros_like(heads)
        mask = tf.constant([[[True, False]], [[True, False]]])
        ripple(ids, heads, relations, tails, mask)
        ripple.entity_embedding.set_weights([tf.constant([[1., 0.], [0., 1.], [2., 0.], [0., 2.]]).numpy()])
        ripple.relation_embedding.set_weights([tf.reshape(tf.eye(2), (1, 4)).numpy()])
        with tf.GradientTape() as tape:
            score = ripple(ids, heads, relations, tails, mask)
        tf.debugging.assert_near(score, tf.sigmoid([10.]))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(tape.gradient(score, ripple.trainable_variables)[0]))))

        kgat = KGAT(4, 1, 2, 2)
        out = kgat(tf.constant([0]), tf.constant([[1, 2]]), tf.constant([[0, 0]]),
                   tf.constant([[True, False]]))
        padded = kgat(tf.constant([0]), tf.constant([[1, 3]]), tf.constant([[0, 0]]),
                      tf.constant([[True, False]]))
        tf.debugging.assert_near(out, padded)
        with tf.GradientTape() as tape:
            result = kgat(tf.constant([0]), tf.constant([[1, 2]]), tf.constant([[0, 0]]),
                          tf.constant([[True, False]]))
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g)))
                            for g in tape.gradient(result, kgat.trainable_variables) if g is not None))
        exact_kgat = KGAT(4, 1, 2, 2)
        exact_kgat(tf.constant([0]), tf.constant([[1, 2]]), tf.constant([[0, 0]]),
                   tf.constant([[True, True]]))
        exact_kgat.entity_embedding.set_weights([tf.constant([[1., 0.], [0., 1.],
                                                               [1., 1.], [0., 0.]]).numpy()])
        exact_kgat.relation_embedding.set_weights([tf.zeros((1, 2)).numpy()])
        exact_kgat.relation_projection.set_weights([tf.reshape(tf.eye(2), (1, 4)).numpy()])
        exact_kgat.sum_projection.set_weights([tf.eye(2).numpy()])
        exact_kgat.product_projection.set_weights([tf.eye(2).numpy()])
        with tf.GradientTape() as tape:
            exact_out = exact_kgat(tf.constant([0]), tf.constant([[1, 2]]),
                                   tf.constant([[0, 0]]), tf.constant([[True, True]]))
        share = tf.sigmoid(tf.tanh(1.))
        tf.debugging.assert_near(exact_out, tf.stack((1 + 2 * share, 1.))[None])
        projection_gradient = tape.gradient(exact_out, exact_kgat.relation_projection.trainable_variables)[0]
        self.assertTrue(bool(tf.reduce_any(tf.abs(projection_gradient) > 0)))

        cdl = CDL(2, 3, 3, 4, 2)
        with tf.GradientTape() as tape:
            cdl_loss = cdl.training_loss(tf.constant([0, 1]), tf.constant([1, 2]),
                                tf.constant([[0., 1., 0.], [1., 0., 0.]]),
                                tf.constant([[0., 1., 1.], [1., 0., 0.]]),
                                tf.constant([1., 0.]), tf.constant([2., 0.2]))
        gradients = tape.gradient(cdl_loss, cdl.trainable_variables)
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients if g is not None))
        self.assertTrue(any(bool(tf.reduce_any(tf.abs(g) > 0)) for g in gradients if g is not None))


if __name__ == "__main__":
    unittest.main()


class GraphFrameworkParityTests(unittest.TestCase):
    def test_fractional_line_edge_weights_preserve_normalized_objective(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.graph import LINE as PT
        from model_evo_harness.models.tensorflow.graph import LINE as TF
        pt, kt = PT(3, 1), TF(3, 1)
        kt(tf.constant([0]), tf.constant([1]))
        with torch.no_grad():
            pt.source_embedding.weight.fill_(1.)
            pt.context_embedding.weight.fill_(1.)
        kt.source_embedding.set_weights([tf.ones((3, 1)).numpy()])
        kt.context_embedding.set_weights([tf.ones((3, 1)).numpy()])
        a = pt.training_loss(torch.tensor([0]), torch.tensor([1]), torch.tensor([[2]]), torch.tensor([.2]))
        b = kt.training_loss(tf.constant([0]), tf.constant([1]), tf.constant([[2]]), tf.constant([.2]))
        self.assertAlmostEqual(a.item(), float(b), places=6)
        self.assertAlmostEqual(a.item(), float(tf.nn.softplus(-1.)+tf.nn.softplus(1.)), places=6)

    def test_kgat_negative_branch_and_tf_graph_agree(self):
        import torch
        import tensorflow as tf
        from model_evo_harness.models.pytorch.graph import KGAT as PT
        from model_evo_harness.models.tensorflow.graph import KGAT as TF
        pt, kt = PT(2, 1, 1, 1), TF(2, 1, 1, 1)
        inputs = (tf.constant([0]), tf.constant([[1]]), tf.constant([[0]]), tf.constant([[True]]))
        kt(*inputs)
        with torch.no_grad():
            pt.entity_embedding.weight.fill_(-1.)
            pt.relation_embedding.weight.zero_()
            pt.relation_projection.weight.fill_(1.)
            pt.sum_projection.weight.fill_(1.)
            pt.product_projection.weight.fill_(-1.)
        kt.entity_embedding.set_weights([-tf.ones((2, 1)).numpy()])
        kt.relation_embedding.set_weights([tf.zeros((1, 1)).numpy()])
        kt.relation_projection.set_weights([tf.ones((1, 1)).numpy()])
        kt.sum_projection.set_weights([tf.ones((1, 1)).numpy()])
        kt.product_projection.set_weights([-tf.ones((1, 1)).numpy()])
        a = pt(torch.tensor([0]), torch.tensor([[1]]), torch.tensor([[0]]), torch.tensor([[True]]))
        b = tf.function(kt)(*inputs)
        self.assertAlmostEqual(a.item(), -.6, places=6)
        self.assertAlmostEqual(a.item(), float(b[0, 0]), places=6)
