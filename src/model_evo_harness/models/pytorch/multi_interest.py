"""Independent native ComiRec interest extractors (https://arxiv.org/abs/2005.09347).

SA follows paper attention with the author's score-only positional option. DR
uses position/interest projections and stopped non-final votes described by
THUDM/ComiRec, src/model.py at 7e605f1. No third-party model code is included.
The host owns embeddings, training objectives, candidate readout and retrieval.
Inputs must be floating history [B,T,D] and bool valid [B,T], True=valid. Valid
values must be finite; masked NaN/Inf padding is sanitized before projection.
"""
import math
from numbers import Integral

import torch
from torch import nn


def _positive(**sizes):
    for name, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _history(history, valid, dim, max_length):
    if history.ndim != 3 or history.shape[-1] != dim or valid.shape != history.shape[:2]:
        raise ValueError("expected history [B,T,dim] and aligned valid [B,T]")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be bool with True=valid")
    if not history.is_floating_point():
        raise TypeError("history must have floating dtype")
    if max_length is not None and history.shape[1] > max_length:
        raise ValueError("history length exceeds max_length")
    return torch.where(valid[..., None], history, 0)


def _squash(values):
    work = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
    squared_norm = work.square().sum(-1, keepdim=True)
    return (squared_norm / (1 + squared_norm) * work / torch.sqrt(squared_norm + 1e-9)).to(values.dtype)


class ComiRecSA(nn.Module):
    """Candidate-independent self-attentive interests.

    Bias-free hidden -> tanh -> score layers produce K logits per event;
    hidden_dim defaults to dim. If max_length is set, trainable position [T,D]
    is added only when computing scores. Values remain the original history.
    Returns interests [B,K,D], interest_mask [B,K], assignments [B,K,T]. Each
    supported interest's assignments sum to one over T. interest_mask reports
    history presence for every K slot, not distinct-cluster usage. Empty or
    all-masked histories return exact zeros and False masks. Position indices
    are physical input slots; the host owns ordering and time-safe truncation.
    """

    def __init__(self, dim: int, num_interests: int, hidden_dim: int | None = None,
                 max_length: int | None = None):
        super().__init__()
        hidden_dim = dim if hidden_dim is None else hidden_dim
        _positive(dim=dim, num_interests=num_interests, hidden_dim=hidden_dim)
        if max_length is not None:
            _positive(max_length=max_length)
        self.dim, self.num_interests, self.max_length = dim, num_interests, max_length
        self.hidden = nn.Linear(dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, num_interests, bias=False)
        self.position = None if max_length is None else nn.Parameter(torch.randn(max_length, dim) * .02)

    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> dict:
        values = _history(history, valid, self.dim, self.max_length)
        score_input = values if self.position is None else values + self.position[:values.shape[1]]
        scores = self.score(torch.tanh(self.hidden(score_input))).transpose(1, 2)
        if scores.dtype in (torch.float16, torch.bfloat16):
            scores = scores.float()
        masked = scores.masked_fill(~valid[:, None], float("-inf"))
        present = valid.any(-1, keepdim=True)
        # A sentinel keeps softmax finite for T=0 or all-masked histories.
        sentinel = torch.where(present[:, None], float("-inf"), 0.).expand(-1, self.num_interests, -1)
        assignments = torch.softmax(torch.cat([masked, sentinel.to(scores.dtype)], -1), -1)[..., :-1].to(values.dtype)
        return dict(interests=assignments @ values,
                    interest_mask=present.expand(-1, self.num_interests),
                    assignments=assignments)


class ComiRecDR(nn.Module):
    """Position-specific capsule routing with deterministic zero initial logits.

    projection[t,k,input,output] is trainable [max_length,K,D,D]. Votes are
    routed by softmax over K for each event, summed over T, then squashed as
    ||s||^2/(1+||s||^2) * s/sqrt(||s||^2+1e-9). Unlike SA, assignments sum to
    one over K at each valid event; invalid events have zero coefficients.
    With stop_gradient_routing=True, non-final votes are detached for both
    capsule sums and agreement updates. The final weighted sum uses live
    votes, so history/projection gradients still flow. False differentiates
    through all routing iterations. Zero logits replace author-code random
    initialization; no sampled logits or length-dependent interest count.

    Returns interests [B,K,D], interest_mask [B,K], assignments [B,K,T]. The
    mask reports any valid history for every K slot, not cluster occupancy.
    Empty or all-masked histories return exact-zero vectors and coefficients.
    """

    def __init__(self, dim: int, num_interests: int, max_length: int,
                 routing_iterations: int = 3, stop_gradient_routing: bool = True):
        super().__init__()
        _positive(dim=dim, num_interests=num_interests, max_length=max_length,
                  routing_iterations=routing_iterations)
        if not isinstance(stop_gradient_routing, bool):
            raise TypeError("stop_gradient_routing must be bool")
        self.dim, self.num_interests, self.max_length = dim, num_interests, max_length
        self.routing_iterations, self.stop_gradient_routing = routing_iterations, stop_gradient_routing
        self.projection = nn.Parameter(torch.randn(max_length, num_interests, dim, dim) / math.sqrt(dim))

    def forward(self, history: torch.Tensor, valid: torch.Tensor) -> dict:
        history = _history(history, valid, self.dim, self.max_length)
        votes = torch.einsum("btd,tkde->bkte", history, self.projection[:history.shape[1]])
        # Accumulate routing in float32 for low-precision callers.
        work = votes.float() if votes.dtype in (torch.float16, torch.bfloat16) else votes
        routing_votes = work.detach() if self.stop_gradient_routing else work
        logits = work.new_zeros(work.shape[:-1])
        for iteration in range(self.routing_iterations):
            assignments = torch.softmax(logits, dim=1) * valid[:, None]
            final = iteration == self.routing_iterations - 1
            active_votes = work if final else routing_votes
            interests = _squash((assignments[..., None] * active_votes).sum(2))
            if not final:
                logits = logits + (routing_votes * interests[:, :, None]).sum(-1)
        return dict(interests=interests.to(votes.dtype),
                    interest_mask=valid.any(-1, keepdim=True).expand(-1, self.num_interests),
                    assignments=assignments.to(votes.dtype))
