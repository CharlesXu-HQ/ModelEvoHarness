"""Single-view QueryFormer query-generation references, arXiv:2609.16548v1 §4.2.

Independent implementation of the paper mechanism, not author source code.
Table 2 specifies user queries over item keys/values. Gate formula, post-LN
residuals, masked-mean item aggregation, independently parameterized per-domain attention
and empty-input behavior below are explicit engineering choices. No Eq.5 Mix,
multi-view tokenization, packed attention, prediction head or training recipe.
All masks are bool with True=valid. Invalid token values are sanitized before
arithmetic; valid values must be finite. Each module is independently usable.
"""

import math

import torch
from torch import nn


def _check(tokens, mask, dim):
    if tokens.ndim != 3 or tokens.shape[-1] != dim or mask.shape != tokens.shape[:2]:
        raise ValueError("expected tokens [B,L,dim] and matching mask [B,L]")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be bool with True=valid")
    return torch.where(mask[..., None], tokens, 0)


def _accumulate(value):
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def _mean(tokens, mask):
    safe = _accumulate(torch.where(mask[..., None], tokens, 0))
    return (safe.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)).to(tokens.dtype)


def _max(tokens, mask):
    masked = torch.where(mask[..., None], tokens, float("-inf"))
    sentinel = tokens.new_full((tokens.shape[0], 1, tokens.shape[-1]), float("-inf"))
    maximum = torch.cat((masked, sentinel), 1).amax(1)
    return torch.where(mask.any(1, keepdim=True), maximum, 0)


class _Attention(nn.Module):
    """Projected multihead attention with a zero fallback for empty key sets."""

    def __init__(self, dim, num_heads):
        super().__init__()
        if dim < 1 or num_heads < 1 or dim % num_heads:
            raise ValueError("positive dim must be divisible by num_heads")
        self.dim, self.num_heads = dim, num_heads
        self.q_proj, self.k_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.v_proj, self.out_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, queries, keys, query_mask, key_mask):
        queries = _check(queries, query_mask, self.dim)
        keys = _check(keys, key_mask, self.dim)
        if queries.shape[0] != keys.shape[0]:
            raise ValueError("query and key batches must match")
        batch, length, _ = queries.shape
        heads, width = self.num_heads, self.dim // self.num_heads
        q = self.q_proj(queries).reshape(batch, length, heads, width).transpose(1, 2)
        k = self.k_proj(keys).reshape(batch, keys.shape[1], heads, width).transpose(1, 2)
        v = self.v_proj(keys).reshape(batch, keys.shape[1], heads, width).transpose(1, 2)
        # A sentinel is available only when all real keys are absent.
        k = torch.cat((k, k.new_zeros((batch, heads, 1, width))), 2)
        v = torch.cat((v, v.new_zeros((batch, heads, 1, width))), 2)
        available = key_mask.any(1, keepdim=True)
        safe_mask = torch.cat((key_mask, ~available), 1)
        scores = _accumulate(q) @ _accumulate(k).transpose(-2, -1) / math.sqrt(width)
        weights = torch.softmax(scores.masked_fill(~safe_mask[:, None, None], float("-inf")), -1)
        values = (weights @ _accumulate(v)).to(queries.dtype)
        output = self.out_proj(values.transpose(1, 2).reshape(batch, length, self.dim))
        return torch.where((query_mask & available)[..., None], output, 0)


class CoTransformerBlock(nn.Module):
    """[B,U,D], [B,I,D], valid masks -> updated user/item tokens.

    Default two layers follow Table 2: each side self-attends, then user reads
    item. Reference choices: LN(x+self_attn(x)); user=LN(user+g*cross), with
    g=sigmoid(W[user,cross]+b). No reverse cross, FFN or dropout is added.
    """

    def __init__(self, dim: int, num_heads: int, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.dim = dim
        self.user_self = nn.ModuleList(_Attention(dim, num_heads) for _ in range(num_layers))
        self.item_self = nn.ModuleList(_Attention(dim, num_heads) for _ in range(num_layers))
        self.user_cross = nn.ModuleList(_Attention(dim, num_heads) for _ in range(num_layers))
        self.user_gate = nn.ModuleList(nn.Linear(2 * dim, dim) for _ in range(num_layers))
        self.user_self_norm = nn.ModuleList(nn.LayerNorm(dim, eps=1e-5) for _ in range(num_layers))
        self.item_self_norm = nn.ModuleList(nn.LayerNorm(dim, eps=1e-5) for _ in range(num_layers))
        self.user_cross_norm = nn.ModuleList(nn.LayerNorm(dim, eps=1e-5) for _ in range(num_layers))

    def forward(self, user_tokens: torch.Tensor, item_tokens: torch.Tensor,
                user_mask: torch.Tensor, item_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        users = _check(user_tokens, user_mask, self.dim)
        items = _check(item_tokens, item_mask, self.dim)
        if users.shape[0] != items.shape[0]:
            raise ValueError("user and item batches must match")
        for index in range(len(self.user_self)):
            users = self.user_self_norm[index](users + self.user_self[index](users, users, user_mask, user_mask))
            items = self.item_self_norm[index](items + self.item_self[index](items, items, item_mask, item_mask))
            users = torch.where(user_mask[..., None], users, 0)
            items = torch.where(item_mask[..., None], items, 0)
            delta = self.user_cross[index](users, items, user_mask, item_mask)
            gate = torch.sigmoid(self.user_gate[index](torch.cat((users, delta), -1)))
            users = self.user_cross_norm[index](users + gate * delta)
            users = torch.where(user_mask[..., None], users, 0)
        return users, items


class QuerySeqCrossAttn(nn.Module):
    """Candidate-conditioned retrieval: [B,I,D], S histories -> [B,S,D], bool [B,S].

    Masked mean of item tokens queries each domain independently. Empty item
    or history sets return zero and False; domain order follows the input list.
    """

    def __init__(self, dim: int, num_heads: int, num_domains: int):
        super().__init__()
        if num_domains < 1:
            raise ValueError("num_domains must be positive")
        self.dim, self.num_domains = dim, num_domains
        self.attentions = nn.ModuleList(_Attention(dim, num_heads) for _ in range(num_domains))

    def forward(self, item_tokens: torch.Tensor, histories: list[torch.Tensor],
                item_mask: torch.Tensor, history_masks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(histories) != self.num_domains or len(history_masks) != self.num_domains:
            raise ValueError("history and mask lists must match num_domains")
        items = _check(item_tokens, item_mask, self.dim)
        query = _mean(items, item_mask)[:, None]
        valid_item = item_mask.any(1, keepdim=True)
        retrieved, valid = [], []
        for attention, history, mask in zip(self.attentions, histories, history_masks):
            retrieved.append(attention(query, history, valid_item, mask)[:, 0])
            valid.append(valid_item[:, 0] & mask.any(1))
        return torch.stack(retrieved, 1), torch.stack(valid, 1)


class SeqQueryCrossAttn(nn.Module):
    """History probes attend to CONTEXT, yielding [B,S,2,D], bool [B,S,2].

    Probes are masked mean then masked max per domain; standard attention Q
    projections supply the only probe projection. Context is K/V (Eq.6).
    This reference adds no residual or normalization after query generation.
    """

    def __init__(self, dim: int, num_heads: int, num_domains: int):
        super().__init__()
        if num_domains < 1:
            raise ValueError("num_domains must be positive")
        self.dim, self.num_domains = dim, num_domains
        self.attentions = nn.ModuleList(_Attention(dim, num_heads) for _ in range(num_domains))

    def forward(self, context_tokens: torch.Tensor, histories: list[torch.Tensor],
                context_mask: torch.Tensor, history_masks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(histories) != self.num_domains or len(history_masks) != self.num_domains:
            raise ValueError("history and mask lists must match num_domains")
        context = _check(context_tokens, context_mask, self.dim)
        queries, valid = [], []
        for attention, history, mask in zip(self.attentions, histories, history_masks):
            history = _check(history, mask, self.dim)
            probes = torch.stack((_mean(history, mask), _max(history, mask)), 1)
            valid_probes = mask.any(1, keepdim=True).expand(-1, 2)
            queries.append(attention(probes, context, valid_probes, context_mask))
            valid.append(valid_probes & context_mask.any(1, keepdim=True))
        return torch.stack(queries, 1), torch.stack(valid, 1)


class QueryFormerQueryBridge(nn.Module):
    """Single-view Eq.2–4 composition; caller owns downstream history reads/Mix.

    Returns queries [B,S,2,D], query_mask [B,S,2], context_tokens [B,U+I+S,D]
    and context_mask [B,U+I+S]. Context order: updated users, items, retrieved
    domains. No H-view replication, token mixer, prediction head or losses.
    """

    def __init__(self, dim: int, num_heads: int, num_domains: int, num_layers: int = 2):
        super().__init__()
        self.co_transformer = CoTransformerBlock(dim, num_heads, num_layers)
        self.retrieval = QuerySeqCrossAttn(dim, num_heads, num_domains)
        self.query_generator = SeqQueryCrossAttn(dim, num_heads, num_domains)

    def forward(self, user_tokens: torch.Tensor, item_tokens: torch.Tensor,
                histories: list[torch.Tensor], user_mask: torch.Tensor,
                item_mask: torch.Tensor, history_masks: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        users, items = self.co_transformer(user_tokens, item_tokens, user_mask, item_mask)
        retrieved, valid = self.retrieval(items, histories, item_mask, history_masks)
        context = torch.cat((users, items, retrieved), 1)
        context_mask = torch.cat((user_mask, item_mask, valid), 1)
        queries, query_mask = self.query_generator(context, histories, context_mask, history_masks)
        return {"queries": queries, "query_mask": query_mask,
                "context_tokens": context, "context_mask": context_mask}
