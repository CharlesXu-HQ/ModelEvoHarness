"""Generic native composition contracts; tensor execution requires a GPU."""

import importlib.util
from pathlib import Path
import unittest


class ParallelBlockImportTests(unittest.TestCase):
    def test_native_composers_load_without_harness_runtime(self):
        for framework, dependency in (("pytorch", "torch"), ("tensorflow", "tensorflow")):
            with self.subTest(framework=framework):
                if importlib.util.find_spec(dependency) is None:
                    continue
                path = (Path(__file__).resolve().parents[1] / "src" / "model_evo_harness"
                        / "models" / framework / "composition.py")
                spec = importlib.util.spec_from_file_location(f"{framework}_composition", path)
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except FileNotFoundError:
                    self.fail(f"Missing native composition module: {path}")
                if framework == "pytorch":
                    import torch
                    self.assertTrue(issubclass(module.ParallelBranches, torch.nn.Module))
                else:
                    import tensorflow as tf
                    self.assertTrue(issubclass(module.ParallelBranches, tf.keras.layers.Layer))


class PyTorchParallelBlocksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("PyTorch is optional")
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Parallel block tensor tests require CUDA")
        from model_evo_harness.models.pytorch.composition import ParallelBranches
        cls.torch = torch
        cls.ParallelBranches = ParallelBranches

    def setUp(self):
        device = self.torch.device("cuda")
        device.__enter__()
        self.addCleanup(device.__exit__, None, None, None)

    def test_routes_heterogeneous_kwargs_by_name_and_accepts_an_existing_head(self):
        torch = self.torch

        class ContextBranch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, 2, bias=False)

            def forward(self, values, offset):
                return self.projection(values) + offset

        class Concat(torch.nn.Module):
            def forward(self, outputs):
                return torch.cat([outputs["wide"], outputs["context"]], dim=-1)

        model = self.ParallelBranches(
            {"wide": torch.nn.Linear(2, 2, bias=False), "context": ContextBranch()}, Concat())
        head = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            for parameter in [*model.parameters(), *head.parameters()]:
                parameter.fill_(1)
        # Reverse the input mapping order: routing must use names, not zip/order.
        inputs = {"context": {"values": torch.tensor([[3., 4., 5.]]),
                              "offset": torch.tensor([[3.]])},
                  "wide": {"input": torch.tensor([[1., 2.]])}}
        representation = model(inputs)
        self.assertEqual(representation.device.type, "cuda")
        torch.testing.assert_close(representation, torch.tensor([[3., 3., 15., 15.]]))
        torch.testing.assert_close(head(representation), torch.tensor([[36.]]))

    def test_both_branches_and_custom_gate_receive_gradients(self):
        torch = self.torch

        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = torch.nn.Parameter(torch.tensor(0.))

            def forward(self, outputs):
                gate = self.logit.sigmoid()
                return gate * outputs["left"] + (1 - gate) * outputs["right"]

        left = torch.nn.Linear(1, 1, bias=False)
        right = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            left.weight.fill_(1)
            right.weight.fill_(2)
        model = self.ParallelBranches({"left": left, "right": right}, Gate())
        inputs = {"left": {"input": torch.tensor([[2.]])},
                  "right": {"input": torch.tensor([[3.]])}}
        output = model(inputs)
        torch.testing.assert_close(output, torch.tensor([[4.]]))
        for branch, expected in (("left", 4.5), ("right", 5.)):
            changed = {name: {"input": values["input"] + (name == branch)}
                       for name, values in inputs.items()}
            torch.testing.assert_close(model(changed), torch.tensor([[expected]]))
        output.sum().backward()
        torch.testing.assert_close(left.weight.grad, torch.tensor([[1.]]))
        torch.testing.assert_close(right.weight.grad, torch.tensor([[1.5]]))
        torch.testing.assert_close(model.fusion.logit.grad, torch.tensor(-1.))

    def test_shared_instances_accumulate_gradients_and_independent_instances_do_not(self):
        torch = self.torch

        class Add(torch.nn.Module):
            def forward(self, outputs):
                return outputs["left"] + outputs["right"]

        for shared in (True, False):
            with self.subTest(shared=shared):
                left = torch.nn.Linear(2, 1, bias=False)
                right = left if shared else torch.nn.Linear(2, 1, bias=False)
                model = self.ParallelBranches({"left": left, "right": right}, Add())
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.fill_(1)
                output = model({"left": {"input": torch.tensor([[2., 3.]])},
                                "right": {"input": torch.tensor([[5., 7.]])}})
                torch.testing.assert_close(output, torch.tensor([[17.]]))
                output.sum().backward()
                self.assertIs(model.branches["left"], left)
                self.assertIs(model.branches["right"], right)
                self.assertEqual(len(list(model.parameters())), 1 if shared else 2)
                torch.testing.assert_close(left.weight.grad,
                                           torch.tensor([[7., 10.]] if shared else [[2., 3.]]))
                if not shared:
                    torch.testing.assert_close(right.weight.grad, torch.tensor([[5., 7.]]))

    def test_rejects_missing_extra_and_misspelled_branch_keys(self):
        torch = self.torch
        model = self.ParallelBranches(
            {"left": torch.nn.Identity(), "right": torch.nn.Identity()}, torch.nn.Identity())
        argument = {"input": torch.ones(1, 2)}
        for names in (("left",), ("left", "right", "extra"), ("left", "rihgt")):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "branch input keys"):
                model({name: argument for name in names})


class TensorFlowParallelBlocksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("tensorflow") is None:
            raise unittest.SkipTest("TensorFlow is optional")
        import tensorflow as tf
        if not tf.config.list_physical_devices("GPU"):
            raise unittest.SkipTest("Parallel block tensor tests require a TensorFlow GPU")
        from model_evo_harness.models.tensorflow.composition import ParallelBranches
        cls.tf = tf
        cls.ParallelBranches = ParallelBranches

    def setUp(self):
        device = self.tf.device("/GPU:0")
        device.__enter__()
        self.addCleanup(device.__exit__, None, None, None)

    def assertTensorEqual(self, actual, expected):
        self.tf.debugging.assert_near(actual, self.tf.constant(expected, dtype=actual.dtype))

    def test_routes_heterogeneous_kwargs_by_name_and_accepts_an_existing_head(self):
        tf = self.tf

        class ContextBranch(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()
                self.projection = tf.keras.layers.Dense(2, use_bias=False, kernel_initializer="ones")

            def call(self, values, offset):
                return self.projection(values) + offset

        class Concat(tf.keras.layers.Layer):
            def call(self, outputs):
                return tf.concat([outputs["wide"], outputs["context"]], axis=-1)

        model = self.ParallelBranches(
            {"wide": tf.keras.layers.Dense(2, use_bias=False, kernel_initializer="ones"),
             "context": ContextBranch()}, Concat())
        head = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")
        inputs = {"context": {"values": tf.constant([[3., 4., 5.]]),
                              "offset": tf.constant([[3.]])},
                  "wide": {"inputs": tf.constant([[1., 2.]])}}
        representation = model(inputs)
        self.assertIn("GPU:0", representation.device)
        self.assertTensorEqual(representation, [[3., 3., 15., 15.]])
        self.assertTensorEqual(head(representation), [[36.]])

    def test_both_branches_and_custom_gate_receive_gradients(self):
        tf = self.tf

        class Gate(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()
                self.logit = self.add_weight(name="logit", shape=(), initializer="zeros")

            def call(self, outputs):
                gate = tf.sigmoid(self.logit)
                return gate * outputs["left"] + (1 - gate) * outputs["right"]

        left = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")
        right = tf.keras.layers.Dense(1, use_bias=False,
                                      kernel_initializer=tf.keras.initializers.Constant(2))
        model = self.ParallelBranches({"left": left, "right": right}, Gate())
        inputs = {"left": {"inputs": tf.constant([[2.]])},
                  "right": {"inputs": tf.constant([[3.]])}}
        with tf.GradientTape() as tape:
            output = model(inputs)
            loss = tf.reduce_sum(output)
        self.assertTensorEqual(output, [[4.]])
        for branch, expected in (("left", 4.5), ("right", 5.)):
            changed = {name: {"inputs": values["inputs"] + float(name == branch)}
                       for name, values in inputs.items()}
            self.assertTensorEqual(model(changed), [[expected]])
        gradients = tape.gradient(loss, [left.kernel, right.kernel, model.fusion.logit])
        for actual, expected in zip(gradients, ([[1.]], [[1.5]], -1.)):
            self.assertIsNotNone(actual)
            self.assertTensorEqual(actual, expected)
        self.assertEqual(len(model.trainable_variables), 3)

    def test_shared_instances_accumulate_gradients_and_independent_instances_do_not(self):
        tf = self.tf

        class Add(tf.keras.layers.Layer):
            def call(self, outputs):
                return outputs["left"] + outputs["right"]

        for shared in (True, False):
            with self.subTest(shared=shared):
                left = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")
                right = left if shared else tf.keras.layers.Dense(
                    1, use_bias=False, kernel_initializer="ones")
                model = self.ParallelBranches({"left": left, "right": right}, Add())
                with tf.GradientTape() as tape:
                    output = model({"left": {"inputs": tf.constant([[2., 3.]])},
                                    "right": {"inputs": tf.constant([[5., 7.]])}})
                    loss = tf.reduce_sum(output)
                self.assertTensorEqual(output, [[17.]])
                self.assertIs(model.branches["left"], left)
                self.assertIs(model.branches["right"], right)
                self.assertEqual(len(model.trainable_variables), 1 if shared else 2)
                gradients = tape.gradient(loss, model.trainable_variables)
                self.assertTensorEqual(gradients[0], [[7.], [10.]] if shared else [[2.], [3.]])
                if not shared:
                    self.assertTensorEqual(gradients[1], [[5.], [7.]])

    def test_training_context_reaches_branches_and_fusion_in_graph_mode(self):
        tf = self.tf

        class TrainingScale(tf.keras.layers.Layer):
            def call(self, inputs, training=None):
                return inputs * (2 if training else 1)

        class TrainingFusion(tf.keras.layers.Layer):
            def call(self, outputs, training=None):
                return (outputs["left"] + outputs["right"]) * (3 if training else 1)

        model = self.ParallelBranches(
            {"left": TrainingScale(), "right": TrainingScale()}, TrainingFusion())

        @tf.function
        def run(inputs, training):
            return model(inputs, training=training)

        inputs = {"left": {"inputs": tf.constant([[1.]])},
                  "right": {"inputs": tf.constant([[2.]])}}
        self.assertTensorEqual(run(inputs, True), [[18.]])
        self.assertTensorEqual(run(inputs, False), [[3.]])

    def test_rejects_missing_extra_and_misspelled_branch_keys(self):
        tf = self.tf
        model = self.ParallelBranches(
            {"left": tf.keras.layers.Identity(), "right": tf.keras.layers.Identity()},
            tf.keras.layers.Identity())
        argument = {"inputs": tf.ones((1, 2))}
        for names in (("left",), ("left", "right", "extra"), ("left", "rihgt")):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "branch input keys"):
                model({name: argument for name in names})


if __name__ == "__main__":
    unittest.main()
