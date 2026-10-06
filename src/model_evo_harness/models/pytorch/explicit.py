"""Native PyTorch building blocks for selected, mixed-field interactions.

Combine existing categorical embeddings with ``NumericFieldEmbedding`` outputs,
then use ``GroupedFM`` to expose separate within/between-group interaction terms.
Inputs must be preprocessed using training-split statistics. These blocks do not
infer semantic groups, bin numeric values, fuse logits, or add first-order terms.
"""

import torch
from torch import nn


def _group_plan(field_count, groups, interactions):
    if type(field_count) is not int or field_count <= 0:
        raise ValueError("field_count must be a positive integer")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("groups must be a nonempty mapping")
    selected, used = {}, set()
    for name, indices in groups.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("group names must be nonempty strings")
        if not isinstance(indices, (list, tuple)) or not indices:
            raise ValueError("each group must contain field indices")
        for index in indices:
            if type(index) is not int or not 0 <= index < field_count:
                raise ValueError("field indices must be integers within field_count")
            if index in used:
                raise ValueError("groups must not repeat or overlap field indices")
            used.add(index)
        selected[name] = tuple(indices)
    if not isinstance(interactions, (list, tuple)) or not interactions:
        raise ValueError("at least one group interaction is required")
    pairs, seen = [], set()
    for pair in interactions:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each interaction must name two groups")
        left, right = pair
        if not all(isinstance(name, str) and name in selected for name in pair):
            raise ValueError("interaction groups must exist in groups")
        if left == right and len(selected[left]) < 2:
            raise ValueError("within-group interaction requires at least two fields")
        key = tuple(sorted(pair))
        if key in seen:
            raise ValueError("duplicate or reversed group interaction")
        seen.add(key)
        pairs.append((left, right))
    return selected, tuple(pairs)


class NumericFieldEmbedding(nn.Module):
    """Encode each real-valued field as ``x_i * v_i``, without discretization.

    ``numeric`` is floating point ``[B,N]``; optional boolean ``present`` has the
    same shape. Missing values (including NaNs) become zero before multiplication.
    An observed zero also has a zero vector, but remains differentiable in x.
    Include separate missing-indicator fields if their effect should be learned.
    """

    def __init__(self, num_fields: int, embedding_dim: int):
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (num_fields, embedding_dim)):
            raise ValueError("num_fields and embedding_dim must be positive integers")
        self.num_fields = num_fields
        self.weight = nn.Parameter(torch.empty(num_fields, embedding_dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, numeric: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        if numeric.ndim != 2 or numeric.shape[1] != self.num_fields:
            raise ValueError("numeric must have shape [batch, num_fields]")
        if not numeric.is_floating_point():
            raise ValueError("numeric must be floating point")
        if present is not None:
            if present.shape != numeric.shape or present.dtype != torch.bool:
                raise ValueError("present must be a boolean mask matching numeric")
            numeric = torch.where(present, numeric, 0.)
        return numeric.unsqueeze(-1) * self.weight.unsqueeze(0)


class GroupedFM(nn.Module):
    """Second-order terms for selected disjoint field groups; ``[B,F,D] -> [B,P]``.

    A pair ``(a,a)`` sums i<j within group a (no self-field products). ``(a,b)``
    sums all products across a and b. Output columns follow ``interactions``;
    a learned fusion layer or ablation mask may consume them separately.
    Groups may select a subset of input fields, but may not overlap. Computation
    takes O(B*(F+P)*D) without constructing the field-pair tensor.
    """

    def __init__(self, field_count: int, groups: dict[str, list[int]],
                 interactions: list[tuple[str, str]]):
        super().__init__()
        self.groups, self.interactions = _group_plan(field_count, groups, interactions)
        self.field_count = field_count

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        if fields.ndim != 3 or fields.shape[1] != self.field_count or fields.shape[2] <= 0:
            raise ValueError("fields must have shape [batch, field_count, positive embedding_dim]")
        if not fields.is_floating_point():
            raise ValueError("fields must be floating point")
        selected = {name: fields[:, indices, :] for name, indices in self.groups.items()}
        sums = {name: values.sum(dim=1) for name, values in selected.items()}
        outputs = []
        for left, right in self.interactions:
            if left == right:
                term = 0.5 * (sums[left].square() - selected[left].square().sum(dim=1))
            else:
                term = sums[left] * sums[right]
            outputs.append(term.sum(dim=-1))
        return torch.stack(outputs, dim=1)
