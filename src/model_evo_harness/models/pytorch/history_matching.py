"""Independent BST, DMIN and DMR history mechanisms.

Reference: reczoo/FuxiCTR 183da1c944ba8930fcdcd54114837c081959d748,
model_zoo/{BST,DMIN,DMR}/src. Native tensor references, not full models.
Host supplies aligned pre-decision item/context embeddings, heads and objectives.
Masks are boolean True=valid; masked NaNs are removed before arithmetic. Empty
or all-masked history readouts are zero, not a fabricated attention observation.
"""
import math

import torch
from torch import nn


def _inputs(target, history, valid, dim, max_len):
    if (target.ndim != 2 or history.ndim != 3 or valid.ndim != 2
            or target.shape != (history.shape[0], dim)
            or history.shape[-1] != dim or valid.shape != history.shape[:2]
            or valid.dtype != torch.bool or history.shape[1] > max_len):
        raise ValueError("expected target [B,D], history [B,L,D], boolean valid [B,L], L <= max_len")
    return torch.where(valid[..., None], history, 0.)


def _weights(scores, valid):
    valid = torch.broadcast_to(valid, scores.shape)
    masked = torch.where(valid, scores, float("-inf"))
    dummy = torch.where(valid.any(-1, keepdim=True), float("-inf"), 0.)
    return torch.softmax(torch.cat([masked, dummy], -1), -1)[..., :-1] * valid


def _last(values, valid):
    b, length, dim = values.shape
    index = torch.where(valid, torch.arange(length, device=values.device)[None] + 1, 0)
    index = torch.cat([index, index.new_zeros(b, 1)], 1).max(1).values
    padded = torch.cat([values.new_zeros(b, 1, dim), values], 1)
    return padded[torch.arange(b, device=values.device), index]


class _SelfHeads(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, sequence, valid, causal):
        b, length, _ = sequence.shape
        shape = (b, length, self.heads, self.dim//self.heads)
        query, key, value = [layer(sequence).reshape(shape).transpose(1, 2)
                             for layer in (self.query, self.key, self.value)]
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.dim//self.heads)
        allowed = valid[:, None, None, :] & valid[:, None, :, None]
        if causal:
            allowed = allowed & torch.ones(length, length, dtype=torch.bool, device=sequence.device).tril()
        weights = _weights(scores, allowed)
        return weights @ value


class _Refiner(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = _SelfHeads(dim, heads)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.ff1, self.ff2 = nn.Linear(dim, 2*dim), nn.Linear(2*dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, values, valid, causal):
        b, length, dim = values.shape
        heads = self.attention(values, valid, causal)
        attended = heads.transpose(1, 2).reshape(b, length, dim)
        states = self.norm1(values + self.out_proj(attended))
        states = self.norm2(states + self.ff2(torch.relu(self.ff1(states))))
        return torch.where(valid[..., None], states, 0.)


class BSTHistoryEncoder(nn.Module):
    """Append candidate token, bidirectionally transform history+candidate.

    Dict: states [B,L,D] for the original history, mask [B,L],
    candidate_state/representation [B,D], has_history [B]. One learned-position
    Transformer block and target-token readout; no source concat/sum pooling,
    causal option or full prediction head. Candidate state is zero when there
    is no history, while host can keep a separate target-only branch.
    """
    def __init__(self, dim, num_heads=2, max_len=128):
        super().__init__()
        if min(dim, num_heads, max_len) < 1 or dim % num_heads:
            raise ValueError("positive sizes and dim divisible by heads required")
        self.dim, self.max_len = dim, max_len
        self.position = nn.Parameter(torch.randn(max_len+1, dim) * .02)
        self.refiner = _Refiner(dim, num_heads)

    def forward(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim, self.max_len)
        length = history.shape[1]
        values = torch.cat([history, target[:, None]], 1) + self.position[:length+1]
        mask = torch.cat([valid, valid.new_ones(valid.shape[0], 1)], 1)
        values = torch.where(mask[..., None], values, 0.)
        states = self.refiner(values, mask, False)
        has_history = valid.any(1)
        representation = torch.where(has_history[:, None], states[:, -1], 0.)
        return dict(representation=representation, candidate_state=representation,
                    states=states[:, :-1], mask=valid, has_history=has_history)


class DMINInterestRefiner(nn.Module):
    """Causal behavior refinement then distinct head-wise target attentions.

    Second causal self-attention splits into H heads; each expands to D,
    adds the refined sequence, applies its own norm/FFN, then DIN-style
    target+position scoring. Dict: refined [B,L,D], head_states [B,H,L,D],
    interests [B,H,D], attention [B,H,L], representation [B,H*D], mask [B,L].
    Independent no-dropout, two-norm refiner; host owns auxiliary transition
    loss/negative sequences. Returned states retain ordered event alignment.
    """
    def __init__(self, dim, num_heads=2, max_len=128):
        super().__init__()
        if min(dim, num_heads, max_len) < 1 or dim % num_heads:
            raise ValueError("positive sizes and dim divisible by heads required")
        self.dim, self.num_heads, self.max_len = dim, num_heads, max_len
        self.refiner = _Refiner(dim, num_heads)
        self.extractor = _SelfHeads(dim, num_heads)
        self.position = nn.Parameter(torch.randn(max_len, dim) * .02)
        self.expansions = nn.ModuleList([nn.Linear(dim//num_heads, dim, bias=False) for _ in range(num_heads)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-5) for _ in range(num_heads)])
        self.ff1 = nn.ModuleList([nn.Linear(dim, 2*dim) for _ in range(num_heads)])
        self.ff2 = nn.ModuleList([nn.Linear(2*dim, dim) for _ in range(num_heads)])
        self.target_query = nn.ModuleList([nn.Linear(2*dim, dim) for _ in range(num_heads)])
        self.score1 = nn.ModuleList([nn.Linear(4*dim, dim) for _ in range(num_heads)])
        self.score2 = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_heads)])

    def forward(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim, self.max_len)
        b, length, _ = history.shape
        refined = self.refiner(history, valid, True)
        heads = self.extractor(refined, valid, True)
        context = torch.cat([target[:, None].expand(-1, length, -1), self.position[:length].expand(b, -1, -1)], -1)
        interests, states, attention = [], [], []
        for i in range(self.num_heads):
            head = self.norms[i](refined + self.expansions[i](heads[:, i]))
            head = head + self.ff2[i](torch.relu(self.ff1[i](head)))
            head = torch.where(valid[..., None], head, 0.)
            query = self.target_query[i](context)
            joined = torch.cat([query, head, query-head, query*head], -1)
            score = self.score2[i](torch.relu(self.score1[i](joined))).squeeze(-1)
            weight = _weights(score, valid)
            interests.append(torch.einsum("bl,bld->bd", weight, head))
            states.append(head); attention.append(weight)
        interests = torch.stack(interests, 1)
        return dict(representation=interests.reshape(b, self.num_heads*self.dim), interests=interests,
                    refined=refined, head_states=torch.stack(states, 1),
                    attention=torch.stack(attention, 1), mask=valid)


class DMRHistoryMatching(nn.Module):
    """Separate context-conditioned user and target-conditioned item matching.

    Optional context [B,L,D] aligns with events; omission means no side context.
    Positional/context scores produce causal prefix user_states, projected and
    matched by dot product with target. Item path scores target/context against
    each event, exposing both normalized readout and sum of raw scores.
    Dict: representation [B,2D], user_match/item_match [B,D], user_states
    [B,L,D], user_score/item_score [B], item_scores/attention [B,L], mask [B,L].
    Host owns separate input/output item embeddings and next-event supervision;
    this core compares target to its supplied D-dimensional user-state space.
    """
    def __init__(self, dim, max_len=128):
        super().__init__()
        if min(dim, max_len) < 1:
            raise ValueError("dim and max_len must be positive")
        self.dim, self.max_len = dim, max_len
        self.position = nn.Parameter(torch.randn(max_len, dim) * .02)
        self.user_query = nn.Linear(2*dim, dim)
        self.item_query = nn.Linear(3*dim, dim)
        self.user_score1, self.user_score2 = nn.Linear(4*dim, dim), nn.Linear(dim, 1)
        self.item_score1, self.item_score2 = nn.Linear(4*dim, dim), nn.Linear(dim, 1)
        self.user_output = nn.Linear(dim, dim)

    def forward(self, target, history, valid, context=None):
        history = _inputs(target, history, valid, self.dim, self.max_len)
        b, length, _ = history.shape
        if context is None:
            context = torch.zeros_like(history)
        elif context.shape != history.shape:
            raise ValueError("context must align with history [B,L,D]")
        context = torch.where(valid[..., None], context, 0.)
        context = torch.cat([context, self.position[:length].expand(b, -1, -1)], -1)
        user_query = torch.relu(self.user_query(context))
        user_join = torch.cat([user_query, history, user_query-history, user_query*history], -1)
        user_scores = self.user_score2(torch.relu(self.user_score1(user_join))).squeeze(-1)
        allowed = valid[:, :, None] & valid[:, None, :] & torch.ones(length, length, dtype=torch.bool, device=history.device).tril()
        weights = _weights(user_scores[:, None].expand(-1, length, -1), allowed)
        user_states = torch.relu(self.user_output(weights @ history))
        user_states = torch.where(valid[..., None], user_states, 0.)
        user_match = _last(user_states, valid)
        item_query = torch.relu(self.item_query(torch.cat([target[:, None].expand(-1, length, -1), context], -1)))
        item_join = torch.cat([item_query, history, item_query-history, item_query*history], -1)
        item_scores = self.item_score2(torch.relu(self.item_score1(item_join))).squeeze(-1)
        item_scores = torch.where(valid, item_scores, 0.)
        attention = _weights(item_scores, valid)
        item_match = torch.einsum("bl,bld->bd", attention, history)
        return dict(representation=torch.cat([user_match, item_match], -1), user_match=user_match,
                    item_match=item_match, user_states=user_states, user_score=(user_match*target).sum(-1),
                    item_score=item_scores.sum(-1), item_scores=item_scores, attention=attention, mask=valid)
