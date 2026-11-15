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
    """Reset-before-affine GRU family; default preserves the original AUGRU."""
    def __init__(self, dim: int, mode='augru'):
        super().__init__()
        self.mode = mode
        self.gates = nn.Linear(2 * dim, 2 * dim)
        self.candidate = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor, previous: torch.Tensor,
                attention: torch.Tensor) -> torch.Tensor:
        if self.mode == 'aigru':
            x = x * attention[:, None]
        reset, update = self.gates(torch.cat([x, previous], dim=-1)).sigmoid().chunk(2, -1)
        proposal = torch.tanh(self.candidate(torch.cat([x, reset * previous], dim=-1)))
        if self.mode == 'agru':
            update = attention[:, None]
        elif self.mode == 'augru':
            update = update * attention[:, None]
        return previous + update * (proposal - previous)


class DIEN(nn.Module):
    """GRU extraction with selectable GRU/AIGRU/AGRU/AUGRU evolution.

    History [B,T,D] uses left-aligned contiguous-prefix boolean masks. Masked
    padding is removed before arithmetic; target and valid events must be finite.
    Default AUGRU retains the original reset-before-affine cell parameterization.
    Inject NextBehaviorObjective to train the exposed interests with next-event
    supervision. Host owns eligible negatives, loss weighting and time cutoff.
    """
    def __init__(self, dim: int, hidden_dim: int, evolution='augru', auxiliary_objective=None):
        super().__init__()
        if evolution not in ('gru', 'aigru', 'agru', 'augru'):
            raise ValueError("evolution must be gru, aigru, agru or augru")
        self.dim = dim
        self.extractor = nn.GRU(dim, dim, batch_first=True)
        self.evolution = _AUGRUCell(dim, evolution)
        self.auxiliary_objective = auxiliary_objective
        self.head = nn.Sequential(nn.Linear(2 * dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, 1))

    def interest_states(self, history, mask):
        """Return [B,T,D] extracted states, zero at padding; T=0 is supported."""
        if (history.ndim != 3 or history.shape[-1] != self.dim or mask.dtype != torch.bool
                or mask.shape != history.shape[:2]):
            raise ValueError("expected history [B,T,D] and boolean mask [B,T]")
        _require_prefix(mask)
        history = torch.where(mask[..., None], history, 0.)
        if not torch.isfinite(history).all().item():
            raise ValueError("valid history must be finite")
        # A trailing dummy permits T=0 without a separate non-differentiable path.
        padded = torch.cat([history, history.new_zeros(history.shape[0], 1, self.dim)], 1)
        interests, _ = self.extractor(padded)
        return torch.where(mask[..., None], interests[:, :-1], 0.)

    def auxiliary_next_event_loss(self, history, mask, negative_history, interests=None):
        """Match state[t] to event[t+1] and negative_history[t+1].

        Negatives align to original event positions [B,T,D], not already shifted.
        Valid transitions need both adjacent mask entries; empty sets yield zero.
        An optional interests tensor must come from this history's forward pass.
        """
        if self.auxiliary_objective is None:
            raise ValueError("inject an auxiliary_objective before requesting next-event loss")
        if negative_history.shape != history.shape:
            raise ValueError("negative_history must align to unshifted history")
        if interests is None:
            interests = self.interest_states(history, mask)
        else:
            if interests.shape != history.shape or mask.dtype != torch.bool or mask.shape != history.shape[:2]:
                raise ValueError("interests/history/mask must align")
            _require_prefix(mask)
        return self.auxiliary_objective(interests[:, :-1], history[:, 1:],
                                        negative_history[:, 1:], mask[:, :-1] & mask[:, 1:])

    def forward(self, target: torch.Tensor, history: torch.Tensor,
                mask: torch.Tensor, return_details=False):
        interests = self.interest_states(history, mask)
        if target.shape != (history.shape[0], self.dim) or not torch.isfinite(target).all().item():
            raise ValueError("finite target [B,D] required")
        attention = _attention(torch.einsum("btd,bd->bt", interests, target), mask)
        state = torch.zeros_like(target)
        for index in range(history.shape[1]):
            evolved = self.evolution(interests[:, index], state, attention[:, index])
            state = torch.where(mask[:, index, None], evolved, state)
        logit = self.head(torch.cat([target, state], dim=-1)).squeeze(-1)
        return dict(logit=logit, interests=interests, attention=attention,
                    representation=state) if return_details else logit


class DSIN(nn.Module):
    """Session attention, BiLSTM and target activation; optional bias encoding.

    bias_encoding adds independent session scalars, within-session position
    scalars and a shared coordinate vector before attention. Bounds are required
    when enabled; default remains content-only within-session attention.
    Empty sessions are excluded from BiLSTM evolution, including internal holes;
    valid sessions retain original order and original bias-encoding indices.
    """

    def __init__(self, dim: int, hidden_dim: int, heads: int, bias_encoding=False,
                 max_sessions=None, max_session_len=None):
        super().__init__()
        self.bias_encoding = bias_encoding
        if bias_encoding:
            if not max_sessions or not max_session_len or min(max_sessions, max_session_len) < 1:
                raise ValueError("positive max_sessions and max_session_len required for bias encoding")
            self.session_bias = nn.Parameter(torch.randn(max_sessions)*.02)
            self.position_bias = nn.Parameter(torch.randn(max_session_len)*.02)
            self.coordinate_bias = nn.Parameter(torch.randn(dim)*.02)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.evolution = nn.LSTM(dim, dim, batch_first=True, bidirectional=True)
        self.evolution_query = nn.Linear(dim, 2 * dim, bias=False)
        self.head = nn.Sequential(nn.Linear(4 * dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, 1))

    def forward(self, target: torch.Tensor, sessions: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        batch, count, length, dim = sessions.shape
        if mask.dtype != torch.bool or mask.shape != sessions.shape[:-1] or min(count, length) < 1:
            raise ValueError("nonempty session axes and boolean aligned mask required")
        sessions = torch.where(mask[..., None], sessions, 0.)
        if not torch.isfinite(sessions).all().item() or not torch.isfinite(target).all().item():
            raise ValueError("target and valid sessions must be finite")
        if self.bias_encoding:
            if count > len(self.session_bias) or length > len(self.position_bias):
                raise ValueError("session/position length exceeds bias encoding bounds")
            sessions = sessions + self.session_bias[:count][None, :, None, None] + self.position_bias[:length][None, None, :, None] + self.coordinate_bias
            sessions = torch.where(mask[..., None], sessions, 0.)
        flat = sessions.reshape(batch * count, length, dim)
        flat_mask = mask.reshape(batch * count, length)
        safe = flat_mask.clone()
        safe[~safe.any(dim=1), 0] = True
        attended, _ = self.self_attention(flat, flat, flat, key_padding_mask=~safe)
        pooled = _mean(attended, flat_mask).reshape(batch, count, dim)
        session_mask = mask.any(dim=-1)
        # Compact valid sessions before the bidirectional recurrence. Otherwise
        # trailing padding changes the backward state of real sessions.
        positions = torch.arange(count, device=sessions.device)[None]
        order = torch.argsort(torch.where(session_mask, positions, positions+count), dim=1)
        compact = pooled.gather(1, order[..., None].expand(-1, -1, dim))
        lengths = session_mask.sum(1).clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(compact, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        encoded, _ = self.evolution(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True,
                                                       total_length=count)
        evolved = torch.zeros_like(encoded).scatter(1, order[..., None].expand(-1, -1, 2*dim), encoded)
        evolved = torch.where(session_mask[..., None], evolved, 0.)
        scores = torch.einsum("bsd,bd->bs", pooled, target)
        evolved_scores = torch.einsum("bsd,bd->bs", evolved,
                                      self.evolution_query(target))
        local = torch.einsum("bs,bsd->bd", _attention(scores, session_mask), pooled)
        temporal = torch.einsum("bs,bsd->bd", _attention(evolved_scores, session_mask),
                                evolved)
        return self.head(torch.cat([target, local, temporal], dim=-1)).squeeze(-1)
