"""Compose arbitrary Keras branches without a model catalog or factory."""

from collections.abc import Mapping
from typing import Any

import tensorflow as tf


class ParallelBranches(tf.keras.layers.Layer):
    """Route named kwargs to branches, then pass their named outputs to fusion.

    Each branch is called as ``branch(**branch_inputs[name])`` and fusion as
    ``fusion(outputs_by_name)``. Inputs and outputs follow those native layers'
    contracts; fusion may return any structure accepted by the downstream head.
    Reusing the same layer instance shares variables; separate instances are
    independent. Keras propagates the outer ``training`` context to nested layer
    calls, including fusion. No layers are cloned or created during ``call``.
    """

    def __init__(self, branches: Mapping[str, tf.keras.layers.Layer],
                 fusion: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        self.branches = dict(branches)
        self.fusion = fusion

    def call(self, branch_inputs: Mapping[str, Mapping[str, Any]],
             training: bool | None = None) -> Any:
        expected, actual = set(self.branches), set(branch_inputs)
        if actual != expected:
            raise ValueError(
                "branch input keys must match registered branches; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
        outputs = {name: branch(**branch_inputs[name])
                   for name, branch in self.branches.items()}
        return self.fusion(outputs)
