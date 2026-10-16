"""Sequence tensor cores; callers own histories, labels, and sampling."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _encoder(dim: int, heads: int) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0,
                                      batch_first=True, norm_first=True)


def _safe(mask: torch.Tensor) -> torch.Tensor:
    safe = mask.clone()
    safe[~mask.any(-1), 0] = True
    return safe


class TransAct(nn.Module):
    """Candidate-conditioned action encoder; recent K states plus masked max.

    ``history [B,T,D], candidate [B,D], mask [B,T] -> [B,(K+1)D]``.
    Per-action candidate concatenation follows the paper's chosen early
    fusion. History is oldest to newest; output starts with the most recent
    valid state. Action-type embeddings and ranking head remain caller-owned.
    """

    def __init__(self, dim: int, num_heads: int, recent_k: int):
        super().__init__()
        if recent_k < 1:
            raise ValueError("recent_k must be positive")
        self.recent_k = recent_k
        self.fuse = nn.Linear(2 * dim, dim)
        self.encoder = _encoder(dim, num_heads)

    def forward(self, history: torch.Tensor, candidate: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if history.shape[1] == 0:
            raise ValueError("history needs at least one slot")
        paired = torch.cat((history, candidate[:, None].expand_as(history)), -1)
        states = self.encoder(self.fuse(paired) * mask.unsqueeze(-1),
                              src_key_padding_mask=~_safe(mask))
        states = states * mask.unsqueeze(-1)
        positions = torch.arange(history.shape[1], device=history.device)[None].expand_as(mask)
        order = positions.masked_fill(~mask, -1).argsort(-1, descending=True)
        k = min(self.recent_k, history.shape[1])
        recent = states.gather(1, order[:, :k, None].expand(-1, -1, states.shape[-1]))
        valid = mask.gather(1, order[:, :k])
        recent = (recent * valid.unsqueeze(-1)).flatten(1)
        if k < self.recent_k:
            recent = F.pad(recent, (0, (self.recent_k - k) * states.shape[-1]))
        maximum = states.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
        maximum = torch.where(mask.any(1, keepdim=True), maximum, 0)
        return torch.cat((recent, maximum), -1)


class PinnerFormer(nn.Module):
    """Compact causal encoder with caller-supplied dense all-action targets.

    ``encode`` returns L2-normalized states [B,T,D]; ``forward`` returns
    L2-normalized last valid [B,D]. This is a compact Transformer core, not
    the full Pinterest training or item-encoder pipeline.
    ``dense_all_action_loss`` consumes preselected positive [B,T,D] and
    negative [B,T,N,D] future item vectors with supervised-position mask
    [B,T]. It neither samples targets nor constructs future time windows.
    """

    def __init__(self, dim: int, num_heads: int, max_len: int):
        super().__init__()
        self.position = nn.Embedding(max_len, dim)
        self.encoder = _encoder(dim, num_heads)

    def encode(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        length = history.shape[1]
        if length == 0:
            raise ValueError("history needs at least one slot")
        x = history * mask.unsqueeze(-1) + self.position(
            torch.arange(length, device=history.device))[None]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                      device=history.device), 1)
        states = self.encoder(x, src_mask=causal,
                              src_key_padding_mask=~_safe(mask))
        return F.normalize(states, dim=-1) * mask.unsqueeze(-1)

    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        states = self.encode(history, mask)
        indices = torch.arange(mask.shape[1], device=mask.device)[None].expand_as(mask)
        last = indices.masked_fill(~mask, -1).max(1).values.clamp_min(0)
        return states[torch.arange(mask.shape[0], device=mask.device), last]

    @staticmethod
    def dense_all_action_loss(states: torch.Tensor, positive: torch.Tensor,
                              negatives: torch.Tensor, supervision: torch.Tensor,
                              temperature: float = 1.0) -> torch.Tensor:
        query = F.normalize(states, dim=-1)
        pos = (query * F.normalize(positive, dim=-1)).sum(-1, keepdim=True)
        neg = torch.einsum("btd,btnd->btn", query, F.normalize(negatives, dim=-1))
        logits = torch.cat((pos, neg), -1) / temperature
        loss = F.cross_entropy(logits.flatten(0, 1),
                               torch.zeros(logits.shape[0] * logits.shape[1], dtype=torch.long,
                                           device=logits.device), reduction="none")
        weight = supervision.flatten().to(loss.dtype)
        return (loss * weight).sum() / weight.sum().clamp_min(1)


class Seq2Slate(nn.Module):
    """Pointer decoder over candidate vectors [B,N,D].

    ``teacher_forced_logits(candidates, eligibility, targets[B,K])`` returns
    [B,K,N] masked logits and rejects ineligible/repeated targets. ``forward``
    greedily returns unique indices [B,K], padded -1 when exhausted.
    """

    def __init__(self, dim: int, slate_size: int):
        super().__init__()
        if slate_size < 1:
            raise ValueError("slate_size must be positive")
        self.slate_size = slate_size
        self.key = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.initial = nn.Linear(dim, dim)
        self.cell = nn.GRUCell(dim, dim)

    def _initial_state(self, candidates: torch.Tensor,
                       eligibility: torch.Tensor) -> torch.Tensor:
        count = eligibility.sum(1, keepdim=True).clamp_min(1)
        context = (candidates * eligibility.unsqueeze(-1)).sum(1) / count
        return torch.tanh(self.initial(context))

    def _scores(self, keys: torch.Tensor, state: torch.Tensor,
                available: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bnd,bd->bn", keys, self.query(state)) / math.sqrt(keys.shape[-1])
        return scores.masked_fill(~available, torch.finfo(scores.dtype).min)

    def teacher_forced_logits(self, candidates: torch.Tensor,
                              eligibility: torch.Tensor,
                              targets: torch.Tensor) -> torch.Tensor:
        batch, _, dim = candidates.shape
        if targets.shape[1] == 0:
            return candidates.sum(-1)[:, None, :][:, :0, :]
        keys = self.key(candidates)
        state = self._initial_state(candidates, eligibility)
        available = eligibility.clone()
        logits = []
        for step in range(targets.shape[1]):
            chosen = targets[:, step]
            if not available.gather(1, chosen[:, None]).all().item():
                raise ValueError("teacher target is ineligible or repeated")
            logits.append(self._scores(keys, state, available))
            state = self.cell(candidates[torch.arange(batch, device=candidates.device), chosen], state)
            available = available.scatter(1, chosen[:, None], False)
        return torch.stack(logits, 1)

    def forward(self, candidates: torch.Tensor, eligibility: torch.Tensor) -> torch.Tensor:
        batch, _, dim = candidates.shape
        if candidates.shape[1] == 0:
            return torch.full((batch, self.slate_size), -1, dtype=torch.long,
                              device=candidates.device)
        keys = self.key(candidates)
        state = self._initial_state(candidates, eligibility)
        available = eligibility.clone()
        result = []
        for _ in range(self.slate_size):
            active = available.any(1)
            index = self._scores(keys, state, available).argmax(1)
            result.append(torch.where(active, index, -1))
            chosen = candidates[torch.arange(batch, device=candidates.device), index]
            state = torch.where(active[:, None], self.cell(chosen, state), state)
            available = available.scatter(1, index[:, None], False)
        return torch.stack(result, 1)
