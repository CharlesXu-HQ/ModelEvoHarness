"""Original local components inspired by context-rich SIF events and CCFormer reads.

Sources: https://arxiv.org/html/2604.15650v3 (§3.1) and
https://arxiv.org/html/2607.28070v2 (§3.1). This implements neither a SIF
quantizer nor a full CCFormer hierarchy. Only torch and stdlib are required.
All masks are bool, True=valid. Valid features must be finite floating tensors;
masked NaN/Inf padding is sanitized before projection and has zero gradient.
"""

import math

import torch
from torch import nn


def _check(tokens, mask, dim):
    if tokens.ndim != 3 or tokens.shape[-1] != dim or mask.shape != tokens.shape[:2]:
        raise ValueError("expected tokens [B,L,dim] and aligned mask [B,L]")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be bool with True=valid")
    if not tokens.is_floating_point():
        raise TypeError("tokens must have floating dtype")
    return torch.where(mask[..., None], tokens, 0)


class ContextualHistoryEncoder(nn.Module):
    """Aligned item [B,L,I], context [B,L,C], mask [B,L] -> tokens, mask.

    tokens[b,t] = tanh(W_item item[b,t] + W_context context[b,t] + bias).
    Output width is dim. Every event keeps its input position; no pooling,
    sorting, positional encoding or history creation is performed. Supply time
    or position in context when temporal distinctions are needed downstream.
    Context fields must be aligned to each event and known before prediction.
    The affine/tanh fusion is an engineering choice, not SIF tokenization.
    """

    def __init__(self, item_dim: int, context_dim: int, dim: int):
        super().__init__()
        if min(item_dim, context_dim, dim) < 1:
            raise ValueError("item_dim, context_dim and dim must be positive")
        self.item_dim, self.context_dim = item_dim, context_dim
        self.item_proj = nn.Linear(item_dim, dim)
        self.context_proj = nn.Linear(context_dim, dim, bias=False)

    def forward(self, item_features: torch.Tensor, context_features: torch.Tensor,
                history_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        items = _check(item_features, history_mask, self.item_dim)
        context = _check(context_features, history_mask, self.context_dim)
        tokens = torch.tanh(self.item_proj(items) + self.context_proj(context))
        return torch.where(history_mask[..., None], tokens, 0), history_mask


class _RelationAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, queries, keys, query_mask, key_mask):
        batch, length, _ = queries.shape
        heads, width = self.num_heads, self.dim // self.num_heads
        q = self.q_proj(queries).reshape(batch, length, heads, width).transpose(1, 2)
        k = self.k_proj(keys).reshape(batch, keys.shape[1], heads, width).transpose(1, 2)
        v = self.v_proj(keys).reshape(batch, keys.shape[1], heads, width).transpose(1, 2)
        # A zero sentinel is visible only when all real keys are absent.
        present = key_mask.any(1, keepdim=True)
        valid_keys = torch.cat((key_mask, ~present), 1)
        k = torch.cat((k, k.new_zeros((batch, heads, 1, width))), 2)
        v = torch.cat((v, v.new_zeros((batch, heads, 1, width))), 2)
        work_q, work_k, work_v = (x.float() if x.dtype in (torch.float16, torch.bfloat16) else x for x in (q, k, v))
        scores = work_q @ work_k.transpose(-2, -1) / math.sqrt(width)
        weights = torch.softmax(scores.masked_fill(~valid_keys[:, None, None], float("-inf")), -1)
        mixed = (weights @ work_v).to(q.dtype).transpose(1, 2).reshape(batch, length, self.dim)
        valid = query_mask & present
        output = torch.where(valid[..., None], self.out_proj(mixed), 0)
        inspected = torch.where(valid[:, None, :, None], weights[..., :-1], 0)
        return output, valid, inspected


class RelationHistoryAttention(nn.Module):
    """Independent user->history, target->history, target->user read branches.

    Inputs: user [B,U,D], history [B,L,D], target [B,T,D] and corresponding
    bool masks [B,U], [B,L], [B,T]. Outputs are a dict with user_history
    [B,U,D], target_history [B,T,D], target_user [B,T,D] and each *_mask.
    return_attention=True adds an attention dict keyed by the same relation
    names, with [B,H,U,L], [B,H,T,L], [B,H,T,U] weights, respectively.
    Each valid query sums to one over its own available keys; absent evidence
    or masked queries return exact-zero representations and weights.

    Relations own separate bias-free Q/K/V/output projections. User and history
    inputs remain unchanged, and target->user reads the original user tokens.
    Targets never attend each other, so separate candidate calls and batching
    agree. "Fixed evidence" does not mean detached gradients: callers control
    freezing/detachment. No residual, FFN, fusion, compression or head is added;
    callers may concatenate these branches with an existing backbone state.
    """

    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        if dim < 1 or num_heads < 1 or dim % num_heads:
            raise ValueError("positive dim must be divisible by positive num_heads")
        self.dim = dim
        self.user_history = _RelationAttention(dim, num_heads)
        self.target_history = _RelationAttention(dim, num_heads)
        self.target_user = _RelationAttention(dim, num_heads)

    def forward(self, user_tokens: torch.Tensor, history_tokens: torch.Tensor,
                target_tokens: torch.Tensor, user_mask: torch.Tensor,
                history_mask: torch.Tensor, target_mask: torch.Tensor,
                return_attention: bool = False) -> dict:
        users = _check(user_tokens, user_mask, self.dim)
        history = _check(history_tokens, history_mask, self.dim)
        targets = _check(target_tokens, target_mask, self.dim)
        if users.shape[0] != history.shape[0] or users.shape[0] != targets.shape[0]:
            raise ValueError("user, history and target batches must match")
        result, attention = {}, {}
        for name, queries, keys, qm, km in (
            ("user_history", users, history, user_mask, history_mask),
            ("target_history", targets, history, target_mask, history_mask),
            ("target_user", targets, users, target_mask, user_mask),
        ):
            result[name], result[name + "_mask"], attention[name] = getattr(self, name)(queries, keys, qm, km)
        if return_attention:
            result["attention"] = attention
        return result
