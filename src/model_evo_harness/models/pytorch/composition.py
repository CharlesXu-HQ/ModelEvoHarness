"""Compose arbitrary PyTorch branches without a model catalog or factory."""

from collections.abc import Mapping
from typing import Any

from torch import nn


class ParallelBranches(nn.Module):
    """Route named kwargs to branches, then pass their named outputs to fusion.

    Each branch is called as ``branch(**branch_inputs[name])`` and fusion as
    ``fusion(outputs_by_name)``. Inputs and outputs follow those native modules'
    contracts; fusion may return any structure accepted by the downstream head.
    Reusing the same module instance shares parameters; separate instances are
    independent. Branch names follow ``nn.ModuleDict`` naming rules. "Parallel"
    describes the model graph, not concurrent CUDA streams.
    """

    def __init__(self, branches: Mapping[str, nn.Module], fusion: nn.Module):
        super().__init__()
        self.branches = nn.ModuleDict(branches)
        self.fusion = fusion

    def forward(self, branch_inputs: Mapping[str, Mapping[str, Any]]) -> Any:
        expected, actual = set(self.branches), set(branch_inputs)
        if actual != expected:
            raise ValueError(
                "branch input keys must match registered branches; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
        outputs = {name: branch(**branch_inputs[name])
                   for name, branch in self.branches.items()}
        return self.fusion(outputs)
