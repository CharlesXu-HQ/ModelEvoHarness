"""Native branches -> custom fusion -> an existing head, with GPU-only execution.

Run after installing the harness and the selected framework:
    python examples/parallel_subnetworks.py pytorch
    python examples/parallel_subnetworks.py tensorflow

The first graph has independent branches with different argument signatures.
The second calls the same branch instance twice, sharing all its parameters.
Concat and gated fusion below are user code, not choices in a model factory.
Both produce width 4, so the existing head can consume either representation.
"""

import argparse


def pytorch_example():
    import torch
    from torch import nn
    from model_evo_harness.models.pytorch.composition import ParallelBranches

    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA GPU; no CPU fallback is used.")

    class ContextProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(3, 4)

        def forward(self, features, context):
            return self.dense(torch.cat([features, context], dim=-1))

    class Concat(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(8, 4)

        def forward(self, outputs):
            return self.projection(torch.cat([outputs["left"], outputs["right"]], dim=-1))

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(8, 4)

        def forward(self, outputs):
            left, right = outputs["left"], outputs["right"]
            weight = self.gate(torch.cat([left, right], dim=-1)).sigmoid()
            return weight * left + (1 - weight) * right

    with torch.device("cuda"):
        existing_head = nn.Linear(4, 1)
        independent = ParallelBranches(
            {"left": nn.Linear(3, 4), "right": ContextProjection()}, Concat())
        independent_logits = existing_head(independent({
            "left": {"input": torch.ones(2, 3)},
            "right": {"features": torch.ones(2, 2), "context": torch.ones(2, 1)},
        }))
        shared_projection = nn.Linear(3, 4)
        shared = ParallelBranches(
            {"left": shared_projection, "right": shared_projection}, Gate())
        shared_logits = existing_head(shared({
            "left": {"input": torch.ones(2, 3)},
            "right": {"input": torch.full((2, 3), 2.)},
        }))
        return {"independent_concat": tuple(independent_logits.shape),
                "shared_gated": tuple(shared_logits.shape)}


def tensorflow_example():
    import tensorflow as tf
    from model_evo_harness.models.tensorflow.composition import ParallelBranches

    if not tf.config.list_physical_devices("GPU"):
        raise SystemExit("This example requires a TensorFlow GPU; no CPU fallback is used.")

    class ContextProjection(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(4)

        def call(self, features, context):
            return self.dense(tf.concat([features, context], axis=-1))

    class Concat(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.projection = tf.keras.layers.Dense(4)

        def call(self, outputs):
            return self.projection(tf.concat([outputs["left"], outputs["right"]], axis=-1))

    class Gate(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.gate = tf.keras.layers.Dense(4, activation="sigmoid")

        def call(self, outputs):
            left, right = outputs["left"], outputs["right"]
            weight = self.gate(tf.concat([left, right], axis=-1))
            return weight * left + (1 - weight) * right

    with tf.device("/GPU:0"):
        existing_head = tf.keras.layers.Dense(1)
        independent = ParallelBranches(
            {"left": tf.keras.layers.Dense(4), "right": ContextProjection()}, Concat())
        independent_logits = existing_head(independent({
            "left": {"inputs": tf.ones((2, 3))},
            "right": {"features": tf.ones((2, 2)), "context": tf.ones((2, 1))},
        }))
        shared_projection = tf.keras.layers.Dense(4)
        shared = ParallelBranches(
            {"left": shared_projection, "right": shared_projection}, Gate())
        shared_logits = existing_head(shared({
            "left": {"inputs": tf.ones((2, 3))},
            "right": {"inputs": tf.fill((2, 3), 2.)},
        }))
        return {"independent_concat": tuple(independent_logits.shape),
                "shared_gated": tuple(shared_logits.shape)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=("pytorch", "tensorflow"))
    args = parser.parse_args()
    print(pytorch_example() if args.framework == "pytorch" else tensorflow_example())
