"""Initial-affinity progressive interaction, inspired by MGDIN §3.2.

Source: https://arxiv.org/html/2603.12586v1 . This independent local block
uses raw initial dot affinities, masked softmax, and a shared token FFN;
it does not reproduce the paper's learned initial projections, unnormalized
interaction, semantic-ID tokenizer, or multiple grouping granularities.
"""

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


class _InteractionLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.projection = nn.Linear(dim, dim, bias=False)
        self.ffn_in = nn.Linear(dim, hidden_dim)
        self.ffn_out = nn.Linear(hidden_dim, dim)
        self.attention_norm = nn.LayerNorm(dim, eps=1e-5)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, tokens, edge_mask):
        logits = torch.matmul(self.query(tokens), self.key(tokens).transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~edge_mask, -torch.inf)
        # A global budget can leave entire rows empty. Never softmax all -inf.
        logits = torch.where(edge_mask.any(-1, keepdim=True), logits, 0)
        weights = torch.softmax(logits, dim=-1) * edge_mask
        context = torch.matmul(weights, self.value(tokens))
        states = self.attention_norm(tokens + self.projection(context))
        return self.ffn_norm(states + self.ffn_out(F.gelu(self.ffn_in(states))))


class ProgressiveInteraction(nn.Module):
    """A host-backbone block: [B,F,D] -> [B,F,D], preserving the field axis.

    ``edge_budgets`` gives one global directed-edge count per layer. Counts
    must be nondecreasing integers in [0,F²]; self-edges count normally and
    are not forced open. Ranking uses X0 @ X0.T / sqrt(D), highest first,
    with row-major tie order, once per input batch and reused across layers.
    This hard ranking is not differentiable; the current Q/K/V and FFN are.

    Optional bool ``valid_mask`` [B,F] excludes padded queries and keys
    before arithmetic, including NaN/Inf padding. Budgets clip to the number
    of valid pairs. Empty rows have zero attention context but retain their
    residual/FFN update; padded fields always return zero. Valid inputs must
    be finite floating point tensors.

    ``interaction_masks`` and ``return_masks=True`` expose bool [B,L,F,F].
    [F²]*L gives dense attention and [k]*L gives fixed sparsity with identical
    parameters: L*(4D² + 2DH + H + 5D), H = hidden_dim (default 4D).
    Computation still forms dense F×F matrices; sparse budgets do not imply
    sparse-kernel speedups. The host supplies embeddings, groups and head.
    """

    def __init__(self, num_fields: int, dim: int, edge_budgets,
                 hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = 4 * dim if hidden_dim is None else hidden_dim
        if any(not isinstance(n, Integral) or isinstance(n, bool) or n < 1
               for n in (num_fields, dim, hidden_dim)):
            raise ValueError("num_fields, dim and hidden_dim must be positive integers")
        try:
            budgets = tuple(edge_budgets)
        except TypeError as exc:
            raise ValueError("edge budgets must be a nonempty integer sequence") from exc
        if (not budgets or any(not isinstance(k, Integral) or isinstance(k, bool) or
                               k < 0 or k > num_fields ** 2 for k in budgets) or
                any(a > b for a, b in zip(budgets, budgets[1:]))):
            raise ValueError("edge budgets must be nondecreasing integers in [0,num_fields²]")
        self.num_fields, self.dim = num_fields, dim
        self.edge_budgets = budgets
        self.blocks = nn.ModuleList([_InteractionLayer(dim, hidden_dim) for _ in budgets])

    def _prepare(self, tokens, valid_mask):
        if tokens.ndim != 3 or tokens.shape[1:] != (self.num_fields, self.dim):
            raise ValueError("tokens must have shape [B,num_fields,dim]")
        if not tokens.is_floating_point():
            raise ValueError("tokens must have a floating point dtype")
        if valid_mask is None:
            valid_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        elif valid_mask.dtype != torch.bool or valid_mask.shape != tokens.shape[:2]:
            raise ValueError("valid_mask must be bool [B,num_fields]")
        safe = torch.where(valid_mask[..., None], tokens, 0)
        if not torch.isfinite(safe).all():
            raise ValueError("valid tokens must be finite")
        return safe, valid_mask

    def _masks(self, tokens, valid_mask):
        initial = tokens.detach()
        affinities = torch.matmul(initial, initial.transpose(-2, -1)) * self.dim ** -0.5
        pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
        flat = affinities.masked_fill(~pairs, -torch.inf).flatten(1)
        order = torch.argsort(flat, dim=-1, descending=True, stable=True)
        ranks = torch.argsort(order, dim=-1).reshape_as(affinities)
        return torch.stack([pairs & (ranks < k) for k in self.edge_budgets], dim=1)

    def interaction_masks(self, tokens: torch.Tensor,
                          valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return the initial-ranking masks [B,L,F,F] without updating tokens."""
        tokens, valid_mask = self._prepare(tokens, valid_mask)
        return self._masks(tokens, valid_mask)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor | None = None,
                return_masks: bool = False):
        states, valid_mask = self._prepare(tokens, valid_mask)
        masks = self._masks(states, valid_mask)
        for i, block in enumerate(self.blocks):
            states = block(states, masks[:, i])
            states = torch.where(valid_mask[..., None], states, 0)
        return (states, masks) if return_masks else states
