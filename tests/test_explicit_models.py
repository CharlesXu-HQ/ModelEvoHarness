"""Mixed-field explicit interaction contracts; model execution requires a GPU."""

import ast
import importlib.util
from pathlib import Path
import unittest


_MODELS = Path(__file__).resolve().parents[1] / "src/model_evo_harness/models"
_GROUPS = {"categorical": [0, 1], "numeric": [2, 3]}
_PAIRS = [("categorical", "categorical"), ("categorical", "numeric"),
          ("numeric", "numeric")]


class ExplicitSourceContractTests(unittest.TestCase):
    def test_both_references_are_standalone_native_framework_modules(self):
        for framework, dependency in [("pytorch", "torch"), ("tensorflow", "tensorflow")]:
            with self.subTest(framework=framework):
                path = _MODELS / framework / "explicit.py"
                self.assertTrue(path.is_file(), f"missing {framework} explicit reference")
                tree = ast.parse(path.read_text())
                classes = {node.name: node for node in tree.body
                           if isinstance(node, ast.ClassDef)}
                self.assertEqual(set(classes), {"NumericFieldEmbedding", "GroupedFM"})
                imports = {name.name.split(".")[0] for node in ast.walk(tree)
                           if isinstance(node, ast.Import) for name in node.names}
                imports.update(node.module.split(".")[0] for node in ast.walk(tree)
                               if isinstance(node, ast.ImportFrom) and node.module)
                self.assertEqual(imports, {dependency})


class PyTorchExplicitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("PyTorch is optional")
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("explicit model tests require CUDA")

    def test_mixed_pairs_equal_direct_expansion_and_have_gradients(self):
        import torch
        from model_evo_harness.models.pytorch.explicit import GroupedFM, NumericFieldEmbedding

        numeric = NumericFieldEmbedding(2, 2).cuda()
        with torch.no_grad():
            numeric.weight.copy_(torch.tensor([[1., 2.], [3., 4.]], device="cuda"))
        values = torch.tensor([[2., -1.], [0., 3.]], device="cuda", requires_grad=True)
        categorical = torch.tensor([[[1., 3.], [2., 4.]], [[2., 1.], [4., 3.]]],
                                   device="cuda", requires_grad=True)
        vectors = torch.cat([categorical, numeric(values)], dim=1)
        actual = GroupedFM(4, _GROUPS, _PAIRS).cuda()(vectors)
        expected = torch.stack([
            (vectors[:, 0] * vectors[:, 1]).sum(-1),
            sum((vectors[:, i] * vectors[:, j]).sum(-1)
                for i in (0, 1) for j in (2, 3)),
            (vectors[:, 2] * vectors[:, 3]).sum(-1),
        ], dim=1)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        for gradient in (numeric.weight.grad, categorical.grad, values.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_missing_values_are_zeroed_without_hiding_observed_zero(self):
        import torch
        from model_evo_harness.models.pytorch.explicit import NumericFieldEmbedding

        model = NumericFieldEmbedding(3, 2).cuda()
        values = torch.tensor([[float("nan"), 0., 2.]], device="cuda", requires_grad=True)
        present = torch.tensor([[False, True, True]], device="cuda")
        result = model(values, present)
        self.assertTrue(torch.isfinite(result).all().item())
        torch.testing.assert_close(result[:, :2], torch.zeros((1, 2, 2), device="cuda"))
        torch.testing.assert_close(result[:, 2], 2 * model.weight[2:3])
        result.sum().backward()
        self.assertEqual(values.grad[0, 0].item(), 0.)
        self.assertGreater(values.grad[0, 1].abs().item(), 0.)
        self.assertEqual(model.weight.grad[0].abs().sum().item(), 0.)

    def test_subset_excludes_unselected_field_and_singleton_cross_is_valid(self):
        import torch
        from model_evo_harness.models.pytorch.explicit import GroupedFM

        vectors = torch.tensor([[[1., 2.], [float("nan"), float("nan")], [3., 4.]]],
                               device="cuda")
        actual = GroupedFM(3, {"left": [0], "right": [2]}, [("left", "right")]).cuda()(vectors)
        torch.testing.assert_close(actual, torch.tensor([[11.]], device="cuda"))

    def test_invalid_contracts_are_rejected(self):
        import torch
        from model_evo_harness.models.pytorch.explicit import GroupedFM, NumericFieldEmbedding

        for groups, pairs in _INVALID_PLANS:
            with self.subTest(groups=groups, pairs=pairs), self.assertRaises(ValueError):
                GroupedFM(4, groups, pairs)
        for count, dim in [(0, 2), (2, 0), (True, 2)]:
            with self.assertRaises(ValueError):
                NumericFieldEmbedding(count, dim)
        encoder = NumericFieldEmbedding(2, 3).cuda()
        with self.assertRaises(ValueError):
            encoder(torch.ones((3, 3), device="cuda"))
        with self.assertRaises(ValueError):
            encoder(torch.ones((3, 2), device="cuda"), torch.ones((3, 1), device="cuda", dtype=torch.bool))
        with self.assertRaises(ValueError):
            encoder(torch.ones((3, 2), device="cuda"), torch.ones((3, 2), device="cuda"))
        fm = GroupedFM(4, _GROUPS, _PAIRS).cuda()
        for shape in [(2, 4), (2, 3, 3), (2, 4, 0)]:
            with self.assertRaises(ValueError):
                fm(torch.ones(shape, device="cuda"))


_INVALID_PLANS = [
    ({}, []),
    ({"a": []}, [("a", "a")]),
    ({"a": [0, 0]}, [("a", "a")]),
    ({"a": [0, 1], "b": [1, 2]}, [("a", "b")]),
    ({"a": [0, 4]}, [("a", "a")]),
    ({"a": [-1, 0]}, [("a", "a")]),
    ({"a": [0, True]}, [("a", "a")]),
    ({"a": [0, 1]}, []),
    ({"a": [0]}, [("a", "a")]),
    ({"a": [0, 1]}, [("a", "missing")]),
    ({"a": [0, 1]}, [("a", "a"), ("a", "a")]),
    ({"a": [0], "b": [1]}, [("a", "b"), ("b", "a")]),
]


class TensorFlowExplicitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("tensorflow") is None:
            raise unittest.SkipTest("TensorFlow is optional")
        import tensorflow as tf
        if not tf.config.list_physical_devices("GPU"):
            raise unittest.SkipTest("explicit model tests require a TensorFlow GPU")

    def test_mixed_pairs_equal_direct_expansion_and_have_gradients(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.explicit import GroupedFM, NumericFieldEmbedding

        with tf.device("/GPU:0"):
            numeric = NumericFieldEmbedding(2, 2)
            numeric.weight.assign([[1., 2.], [3., 4.]])
            values = tf.Variable([[2., -1.], [0., 3.]])
            categorical = tf.Variable([[[1., 3.], [2., 4.]], [[2., 1.], [4., 3.]]])
            with tf.GradientTape() as tape:
                vectors = tf.concat([categorical, numeric(values)], axis=1)
                actual = GroupedFM(4, _GROUPS, _PAIRS)(vectors)
                loss = tf.reduce_mean(tf.square(actual))
            expected = tf.stack([
                tf.reduce_sum(vectors[:, 0] * vectors[:, 1], -1),
                sum(tf.reduce_sum(vectors[:, i] * vectors[:, j], -1)
                    for i in (0, 1) for j in (2, 3)),
                tf.reduce_sum(vectors[:, 2] * vectors[:, 3], -1),
            ], axis=1)
            tf.debugging.assert_near(actual, expected)
            self.assertIn("GPU", actual.device)
            for gradient in tape.gradient(loss, [numeric.weight, categorical, values]):
                self.assertIsNotNone(gradient)
                self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))
                self.assertGreater(float(tf.reduce_sum(tf.abs(gradient))), 0)

    def test_missing_values_are_zeroed_without_hiding_observed_zero(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.explicit import NumericFieldEmbedding

        with tf.device("/GPU:0"):
            model = NumericFieldEmbedding(3, 2)
            values = tf.Variable([[float("nan"), 0., 2.]])
            present = tf.constant([[False, True, True]])
            with tf.GradientTape() as tape:
                result = model(values, present=present)
                loss = tf.reduce_sum(result)
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(result))))
            tf.debugging.assert_near(result[:, :2], tf.zeros((1, 2, 2)))
            tf.debugging.assert_near(result[:, 2], 2 * model.weight[2:3])
            value_gradient, weight_gradient = tape.gradient(loss, [values, model.weight])
            self.assertEqual(float(value_gradient[0, 0]), 0.)
            self.assertGreater(abs(float(value_gradient[0, 1])), 0.)
            self.assertEqual(float(tf.reduce_sum(tf.abs(weight_gradient[0]))), 0.)

    def test_subset_excludes_unselected_field_and_singleton_cross_is_valid(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.explicit import GroupedFM

        with tf.device("/GPU:0"):
            vectors = tf.constant([[[1., 2.], [float("nan"), float("nan")], [3., 4.]]])
            actual = GroupedFM(3, {"left": [0], "right": [2]}, [("left", "right")])(vectors)
            tf.debugging.assert_near(actual, tf.constant([[11.]]))

    def test_invalid_contracts_are_rejected(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.explicit import GroupedFM, NumericFieldEmbedding

        with tf.device("/GPU:0"):
            for groups, pairs in _INVALID_PLANS:
                with self.subTest(groups=groups, pairs=pairs), self.assertRaises(ValueError):
                    GroupedFM(4, groups, pairs)
            for count, dim in [(0, 2), (2, 0), (True, 2)]:
                with self.assertRaises(ValueError):
                    NumericFieldEmbedding(count, dim)
            encoder = NumericFieldEmbedding(2, 3)
            error_types = (ValueError, tf.errors.InvalidArgumentError)
            with self.assertRaises(error_types):
                encoder(tf.ones((3, 3)))
            with self.assertRaises(error_types):
                encoder(tf.ones((3, 2)), present=tf.ones((3, 1), dtype=tf.bool))
            with self.assertRaises(error_types):
                encoder(tf.ones((3, 2)), present=tf.ones((3, 2)))
            fm = GroupedFM(4, _GROUPS, _PAIRS)
            for shape in [(2, 4), (2, 3, 3), (2, 4, 0)]:
                with self.assertRaises(error_types):
                    fm(tf.ones(shape))


if __name__ == "__main__":
    unittest.main()
