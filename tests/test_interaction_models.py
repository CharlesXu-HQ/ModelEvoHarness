"""Contract tests for direct-framework feature-interaction references."""

import importlib.util
import unittest


_CARDINALITIES = [5, 6, 7]


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class PyTorchInteractionTests(unittest.TestCase):
    def test_all_models_produce_logits_and_gradients(self):
        import torch
        from model_evo_harness.models.pytorch.interactions import (
            AFM, AutoInt, DCNv2, FiBiNET, FwFM, NFM, PNN, WideDeep, XDeepFM,
        )

        torch.manual_seed(7)
        fields = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
        dense = torch.randn(4, 7)
        models = [
            (AFM(_CARDINALITIES, 4, attention_dim=6), fields),
            (AutoInt(_CARDINALITIES, 4, num_heads=2, num_layers=2), fields),
            (FiBiNET(_CARDINALITIES, 4, hidden_dim=8), fields),
            (NFM(_CARDINALITIES, 4, hidden_dim=8), fields),
            (PNN(_CARDINALITIES, 4, hidden_dim=8), fields),
            (WideDeep(_CARDINALITIES, 4, hidden_dim=8,
                      cross_pairs=[(0, 1), (1, 2)]), fields),
            (XDeepFM(_CARDINALITIES, 4, hidden_dim=8, cin_channels=[4, 4]), fields),
            (DCNv2(7, num_cross_layers=2, hidden_dim=8, rank=3), dense),
            (FwFM(_CARDINALITIES, 4), fields),
        ]
        for model, inputs in models:
            with self.subTest(model=type(model).__name__):
                output = model(inputs)
                self.assertEqual(tuple(output.shape), (4,))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(any(parameter.grad is not None and
                                    torch.isfinite(parameter.grad).all()
                                    for parameter in model.parameters()))

    def test_mechanism_properties(self):
        import torch
        from model_evo_harness.models.pytorch.interactions import (
            AFM, AutoInt, DCNv2, FiBiNET, FwFM, NFM, PNN, WideDeep, XDeepFM,
        )

        fields = torch.tensor([[1, 2, 3], [0, 1, 2]])
        weights = AFM(_CARDINALITIES, 4).attention_weights(fields)
        self.assertEqual(tuple(weights.shape), (2, 3))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2)))
        gates = FiBiNET(_CARDINALITIES, 4, hidden_dim=8).field_gates(fields)
        self.assertEqual(tuple(gates.shape), (2, 3))
        self.assertTrue(bool(((gates >= 0) & (gates <= 1)).all()))
        values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        self.assertTrue(torch.allclose(NFM.bi_interaction(values),
                                       torch.tensor([[23.0, 44.0]])))
        products = PNN(_CARDINALITIES, 2, hidden_dim=8).product_features(values)
        self.assertEqual(products[0, 6:9].tolist(), [11.0, 17.0, 39.0])
        self.assertEqual(products[0, 9:13].tolist(), [3.0, 4.0, 6.0, 8.0])
        attended = AutoInt(_CARDINALITIES, 4).interaction_features(fields)
        self.assertEqual(tuple(attended.shape), (2, 3, 4))
        self.assertFalse(torch.allclose(attended[0], attended[1]))
        wide = WideDeep(_CARDINALITIES, 4, 8, cross_pairs=[(0, 1), (1, 2)])
        self.assertEqual(wide.cross_indices(fields).tolist(), [[8, 17], [1, 9]])
        cin = XDeepFM(_CARDINALITIES, 4, 8, cin_channels=[4, 4])
        self.assertEqual(tuple(cin.cin_features(fields).shape), (2, 8))
        fwfm = FwFM(_CARDINALITIES, 4)
        with torch.no_grad():
            fwfm.pair_weights.zero_()
        self.assertTrue(torch.allclose(fwfm(fields), fwfm.first_order(fields)))
        cross = DCNv2(7, 2, 8, rank=3)
        with torch.no_grad():
            for layer in cross.cross_layers:
                for parameter in layer.parameters():
                    parameter.zero_()
        dense = torch.randn(2, 7)
        self.assertTrue(torch.allclose(cross.cross_features(dense), dense))


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
class TensorFlowInteractionTests(unittest.TestCase):
    def test_all_models_produce_logits_and_gradients(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.interactions import (
            AFM, AutoInt, DCNv2, FiBiNET, FwFM, NFM, PNN, WideDeep, XDeepFM,
        )

        tf.random.set_seed(7)
        fields = tf.constant([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
        dense = tf.random.normal((4, 7))
        models = [
            (AFM(_CARDINALITIES, 4, attention_dim=6), fields),
            (AutoInt(_CARDINALITIES, 4, num_heads=2, num_layers=2), fields),
            (FiBiNET(_CARDINALITIES, 4, hidden_dim=8), fields),
            (NFM(_CARDINALITIES, 4, hidden_dim=8), fields),
            (PNN(_CARDINALITIES, 4, hidden_dim=8), fields),
            (WideDeep(_CARDINALITIES, 4, hidden_dim=8,
                      cross_pairs=[(0, 1), (1, 2)]), fields),
            (XDeepFM(_CARDINALITIES, 4, hidden_dim=8, cin_channels=[4, 4]), fields),
            (DCNv2(7, num_cross_layers=2, hidden_dim=8, rank=3), dense),
            (FwFM(_CARDINALITIES, 4), fields),
        ]
        for model, inputs in models:
            with self.subTest(model=type(model).__name__):
                with tf.GradientTape() as tape:
                    output = model(inputs)
                    loss = tf.reduce_mean(tf.square(output))
                self.assertEqual(tuple(output.shape), (4,))
                self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(output))))
                gradients = tape.gradient(loss, model.trainable_variables)
                self.assertTrue(any(g is not None and
                                    bool(tf.reduce_all(tf.math.is_finite(g)))
                                    for g in gradients))

    def test_mechanism_properties(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.interactions import (
            AFM, AutoInt, DCNv2, FiBiNET, FwFM, NFM, PNN, WideDeep, XDeepFM,
        )

        fields = tf.constant([[1, 2, 3], [0, 1, 2]])
        weights = AFM(_CARDINALITIES, 4).attention_weights(fields)
        self.assertEqual(tuple(weights.shape), (2, 3))
        self.assertTrue(bool(tf.reduce_all(tf.abs(tf.reduce_sum(weights, 1) - 1) < 1e-6)))
        gates = FiBiNET(_CARDINALITIES, 4, hidden_dim=8).field_gates(fields)
        self.assertEqual(tuple(gates.shape), (2, 3))
        self.assertTrue(bool(tf.reduce_all((gates >= 0) & (gates <= 1))))
        values = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        self.assertTrue(bool(tf.reduce_all(tf.abs(
            NFM.bi_interaction(values) - tf.constant([[23.0, 44.0]])) < 1e-6)))
        products = PNN(_CARDINALITIES, 2, hidden_dim=8).product_features(values)
        self.assertEqual(products.numpy()[0, 6:9].tolist(), [11.0, 17.0, 39.0])
        self.assertEqual(products.numpy()[0, 9:13].tolist(), [3.0, 4.0, 6.0, 8.0])
        attended = AutoInt(_CARDINALITIES, 4).interaction_features(fields)
        self.assertEqual(tuple(attended.shape), (2, 3, 4))
        self.assertFalse(bool(tf.reduce_all(tf.abs(attended[0] - attended[1]) < 1e-6)))
        wide = WideDeep(_CARDINALITIES, 4, 8, cross_pairs=[(0, 1), (1, 2)])
        self.assertEqual(wide.cross_indices(fields).numpy().tolist(), [[8, 17], [1, 9]])
        cin = XDeepFM(_CARDINALITIES, 4, 8, cin_channels=[4, 4])
        self.assertEqual(tuple(cin.cin_features(fields).shape), (2, 8))
        fwfm = FwFM(_CARDINALITIES, 4)
        fwfm(fields)
        fwfm.pair_weights.assign(tf.zeros_like(fwfm.pair_weights))
        self.assertTrue(bool(tf.reduce_all(tf.abs(fwfm(fields) -
                                                   fwfm.first_order(fields)) < 1e-6)))
        cross = DCNv2(7, 2, 8, rank=3)
        dense = tf.random.normal((2, 7))
        cross(dense)
        for layer in cross.cross_layers:
            for variable in layer.trainable_variables:
                variable.assign(tf.zeros_like(variable))
        self.assertTrue(bool(tf.reduce_all(tf.abs(cross.cross_features(dense) -
                                                   dense) < 1e-6)))


if __name__ == "__main__":
    unittest.main()
