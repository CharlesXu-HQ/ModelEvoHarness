"""Direct PyTorch sequence and interest-model cores.

Inputs are pre-decision embeddings; the host owns IDs, labels, loss, and evaluation.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def _last(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(values.shape[1], device=values.device)[None]
    index = torch.where(mask, positions, -1).max(dim=1).values.clamp_min(0)
    selected = values[torch.arange(values.shape[0], device=values.device), index]
    return selected * mask.any(dim=1, keepdim=True)


def _require_prefix(mask: torch.Tensor) -> None:
    """Recurrent encoders accept left-aligned valid events followed by padding."""
    if torch.any(mask[:, 1:] & ~mask[:, :-1]).item():
        raise ValueError("recurrent history mask must be a contiguous prefix")


def _attention(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(scores.masked_fill(~mask, -1e9), dim=-1) * mask
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)


class MIND(nn.Module):
    """Dynamic routing from history embeddings [B,T,D] to K interest vectors."""

    def __init__(self, dim: int, num_interests: int, routing_iters: int = 3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_interests, dim) / math.sqrt(dim))
        self.routing_iters = routing_iters

    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("btd,kd->btk", history, self.prototypes) / math.sqrt(
            history.shape[-1])
        for _ in range(self.routing_iters):
            assignment = torch.softmax(logits, dim=-1) * mask.unsqueeze(-1)
            interests = torch.einsum("btk,btd->bkd", assignment, history)
            interests = interests / assignment.sum(dim=1).unsqueeze(-1).clamp_min(1)
            interests = F.normalize(interests + self.prototypes.unsqueeze(0), dim=-1)
            logits = logits + torch.einsum("btd,bkd->btk", history, interests)
        return interests

    @staticmethod
    def score(interests: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Max interest-to-item dot product for item vectors [I,D]."""
        return torch.einsum("bkd,id->bki", interests, items).max(dim=1).values

    @staticmethod
    def label_aware_weights(interests: torch.Tensor, target: torch.Tensor,
                            temperature: float = 1.0) -> torch.Tensor:
        """Training-only target-conditioned weights over K interests [B,K,D]."""
        return torch.softmax(torch.einsum("bkd,bd->bk", interests, target) / temperature,
                             dim=1)

    def training_readout(self, history: torch.Tensor, mask: torch.Tensor,
                         target: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Return a target-aware user vector; inference can retain all K interests."""
        interests = self(history, mask)
        weights = self.label_aware_weights(interests, target, temperature)
        return torch.einsum("bk,bkd->bd", weights, interests)


class SASRec(nn.Module):
    """Causal self-attention for ordered history embeddings [B,T,D]."""

    def __init__(self, dim: int, num_heads: int, max_len: int):
        super().__init__()
        self.position = nn.Embedding(max_len, dim)
        self.layer = nn.TransformerEncoderLayer(
            dim, num_heads, 2 * dim, dropout=0, batch_first=True, norm_first=True)

    def encode(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        length = history.shape[1]
        x = history + self.position(torch.arange(length, device=history.device))[None]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                      device=history.device), diagonal=1)
        safe = mask.clone()
        safe[~mask.any(dim=1), 0] = True
        return self.layer(x, src_mask=causal, src_key_padding_mask=~safe) * mask.unsqueeze(-1)

    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _last(self.encode(history, mask), mask)


class SDM(nn.Module):
    """Short LSTM/self-attention and long user-attention with gated fusion.

    Histories are [B,T,D] with contiguous-prefix masks. Omitted long history
    reuses short history; omitted user context is the masked long-history mean.
    """

    def __init__(self, dim: int, short_window: int):
        super().__init__()
        self.short_window = short_window
        heads = 2 if dim % 2 == 0 else 1
        self.short_lstm = nn.LSTM(dim, dim, batch_first=True)
        self.short_attention = nn.MultiheadAttention(dim, heads, dropout=0,
                                                    batch_first=True)
        self.short_query = nn.Linear(dim, dim, bias=False)
        self.long_query = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(3 * dim, dim)

    def forward(self, short_history: torch.Tensor, short_mask: torch.Tensor,
                long_history: torch.Tensor | None = None,
                long_mask: torch.Tensor | None = None,
                user_context: torch.Tensor | None = None) -> torch.Tensor:
        if long_history is None:
            long_history, long_mask = short_history, short_mask
        if long_mask is None:
            raise ValueError("long_mask is required with long_history")
        _require_prefix(short_mask)
        _require_prefix(long_mask)
        if user_context is None:
            user_context = _mean(long_history, long_mask)

        window = min(self.short_window, short_history.shape[1])
        lengths = short_mask.long().sum(dim=1)
        offsets = torch.arange(window, device=short_history.device)[None]
        indices = (lengths - window).clamp_min(0)[:, None] + offsets
        recent = short_history.gather(1, indices.unsqueeze(-1).expand(-1, -1,
                                                                       short_history.shape[-1]))
        recent_mask = offsets < lengths.clamp_max(window)[:, None]
        recent = recent * recent_mask.unsqueeze(-1)
        short_states, _ = self.short_lstm(recent)
        safe = recent_mask.clone()
        safe[~safe.any(dim=1), 0] = True
        attended, _ = self.short_attention(short_states, short_states, short_states,
                                           key_padding_mask=~safe)
        short_scores = torch.einsum("btd,bd->bt", attended,
                                    self.short_query(user_context)) / math.sqrt(short_history.shape[-1])
        short = torch.einsum("bt,btd->bd", _attention(short_scores, recent_mask), attended)
        long_scores = torch.einsum("btd,bd->bt", long_history,
                                   self.long_query(user_context)) / math.sqrt(long_history.shape[-1])
        long = torch.einsum("bt,btd->bd", _attention(long_scores, long_mask), long_history)
        gate = torch.sigmoid(self.gate(torch.cat([short, long, user_context], dim=-1)))
        output = gate * short + (1 - gate) * long
        return output * (short_mask.any(dim=1) | long_mask.any(dim=1))[:, None]


class NARM(nn.Module):
    """GRU session encoder with final-state-conditioned local attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.weight = nn.Linear(dim, 1, bias=False)
        self.fuse = nn.Linear(2 * dim, dim)

    def forward(self, session: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _require_prefix(mask)
        states, _ = self.gru(session * mask.unsqueeze(-1))
        global_state = _last(states, mask)
        scores = self.weight(torch.sigmoid(self.key(states) +
                                           self.query(global_state)[:, None])).squeeze(-1)
        local = torch.einsum("bt,btd->bd", _attention(scores, mask), states)
        return self.fuse(torch.cat([global_state, local], dim=-1)) * mask.any(dim=1)[:, None]


class HSTU(nn.Module):
    """Causal gated pointwise sequence unit with relative-time attention bias."""

    def __init__(self, dim: int, max_time_gap: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkvu = nn.Linear(dim, 4 * dim)
        self.time_bias = nn.Embedding(max_time_gap + 1, 1)
        self.out = nn.Linear(dim, dim)
        self.max_time_gap = max_time_gap

    def forward(self, history: torch.Tensor, timestamps: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        q, k, v, u = self.qkvu(self.norm(history)).chunk(4, dim=-1)
        logits = q @ k.transpose(1, 2) / math.sqrt(history.shape[-1])
        gap = (timestamps[:, :, None] - timestamps[:, None, :]).clamp(
            min=0, max=self.max_time_gap).long()
        logits = logits + self.time_bias(gap).squeeze(-1)
        time = history.shape[1]
        causal = torch.tril(torch.ones(time, time, dtype=torch.bool,
                                      device=history.device))[None]
        valid = causal & mask[:, :, None] & mask[:, None, :]
        weights = F.silu(logits) * valid / math.sqrt(time)
        output = history + self.out((weights @ v) * torch.sigmoid(u))
        return output * mask.unsqueeze(-1)


class _AUGRUCell(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gates = nn.Linear(2 * dim, 2 * dim)
        self.candidate = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor, previous: torch.Tensor,
                attention: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat([x, previous], dim=-1)).sigmoid().chunk(2, -1)
        proposal = torch.tanh(self.candidate(torch.cat([x, reset * previous], dim=-1)))
        update = update * attention[:, None]
        return previous + update * (proposal - previous)


class DIEN(nn.Module):
    """GRU interest extraction and target-aware AUGRU interest evolution."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.extractor = nn.GRU(dim, dim, batch_first=True)
        self.evolution = _AUGRUCell(dim)
        self.head = nn.Sequential(nn.Linear(2 * dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, 1))

    def forward(self, target: torch.Tensor, history: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        interests, _ = self.extractor(history)
        attention = _attention(torch.einsum("btd,bd->bt", interests, target), mask)
        state = torch.zeros_like(target)
        for index in range(history.shape[1]):
            evolved = self.evolution(interests[:, index], state, attention[:, index])
            state = torch.where(mask[:, index, None], evolved, state)
        return self.head(torch.cat([target, state], dim=-1)).squeeze(-1)


class DSIN(nn.Module):
    """Session self-attention, cross-session BiLSTM, and target activation."""

    def __init__(self, dim: int, hidden_dim: int, heads: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.evolution = nn.LSTM(dim, dim, batch_first=True, bidirectional=True)
        self.evolution_query = nn.Linear(dim, 2 * dim, bias=False)
        self.head = nn.Sequential(nn.Linear(4 * dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, 1))

    def forward(self, target: torch.Tensor, sessions: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        batch, count, length, dim = sessions.shape
        flat = sessions.reshape(batch * count, length, dim)
        flat_mask = mask.reshape(batch * count, length)
        safe = flat_mask.clone()
        safe[~safe.any(dim=1), 0] = True
        attended, _ = self.self_attention(flat, flat, flat, key_padding_mask=~safe)
        pooled = _mean(attended, flat_mask).reshape(batch, count, dim)
        evolved, _ = self.evolution(pooled)
        session_mask = mask.any(dim=-1)
        scores = torch.einsum("bsd,bd->bs", pooled, target)
        evolved_scores = torch.einsum("bsd,bd->bs", evolved,
                                      self.evolution_query(target))
        local = torch.einsum("bs,bsd->bd", _attention(scores, session_mask), pooled)
        temporal = torch.einsum("bs,bsd->bd", _attention(evolved_scores, session_mask),
                                evolved)
        return self.head(torch.cat([target, local, temporal], dim=-1)).squeeze(-1)
