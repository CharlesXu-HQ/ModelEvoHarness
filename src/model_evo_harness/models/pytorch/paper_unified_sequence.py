"""Content-only UniDot conditional gates and multi-channel sequence pooling.

Adapted from linrongc/kdd26_unidot at
1399702006a1fc2f86329c8c6133f3f3ac1e6d0d: common.py CondGatedSwiGLU and
unified_model.py MultiChannelSeqPool. Changes: True=valid masks, sanitized
padding, zero empty domains after affine transforms, stable half-type L2 with
zero-direction gradients, internal condition broadcast, and removal of
temporal/interaction variants and data dependencies.

MIT License
Copyright (c) 2026 Rongcheng Lin and the UniDot authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalSwiGLU(nn.Module):
    """History [B,L,D_s], condition [B,D_c], bool valid mask [B,L] -> [B,L,D_o].

    out = Wo(value(history) * SiLU(gate([history, condition]))).
    The condition never enters the value projection. Padding returns zero.
    """

    def __init__(self, sequence_dim: int, condition_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        if min(sequence_dim, condition_dim, hidden_dim, output_dim) < 1:
            raise ValueError("dimensions must be positive")
        self.sequence_dim, self.condition_dim = sequence_dim, condition_dim
        self.value = nn.Linear(sequence_dim, hidden_dim)
        self.gate = nn.Linear(sequence_dim + condition_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, history: torch.Tensor, condition: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        if (history.ndim != 3 or history.shape[-1] != self.sequence_dim or
                condition.shape != (history.shape[0], self.condition_dim) or
                valid_mask.shape != history.shape[:2] or valid_mask.dtype != torch.bool):
            raise ValueError("incompatible history/condition/mask shapes or mask dtype")
        safe = torch.where(valid_mask[..., None], history, 0)
        context = condition[:, None].expand(-1, history.shape[1], -1)
        gate = F.silu(self.gate(torch.cat((safe, context), -1)))
        output = self.output_projection(self.value(safe) * gate)
        return torch.where(valid_mask[..., None], output, 0)


class MultiChannelSequencePool(nn.Module):
    """State [B,T,D], S histories [B,L_s,D], S bool valid masks -> [B,S*C,D].

    Each domain owns a GELU scorer and post-pool projection. Sigmoid weights
    are independent across positions and channels. Weighted sums are L2
    normalized before projection and shared LayerNorm. Empty domains (including
    L=0) return zero after all affine operations. State tokens are fully defined.
    """

    def __init__(self, dim: int, num_state_tokens: int, num_channels: int, num_domains: int):
        super().__init__()
        if min(dim, num_state_tokens, num_channels, num_domains) < 1:
            raise ValueError("dimensions and domain count must be positive")
        self.dim, self.num_state_tokens = dim, num_state_tokens
        self.num_domains, self.num_channels = num_domains, num_channels
        self.state_projection = nn.Linear(num_state_tokens * dim, dim)
        self.scorers = nn.ModuleList(nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(),
                                                   nn.Linear(dim, num_channels)) for _ in range(num_domains))
        self.pool_projections = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_domains))
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, state_tokens: torch.Tensor, histories: list[torch.Tensor],
                valid_masks: list[torch.Tensor]) -> torch.Tensor:
        if len(histories) != self.num_domains or len(valid_masks) != self.num_domains:
            raise ValueError("history/mask lists must match the configured domain count")
        if state_tokens.ndim != 3 or state_tokens.shape[1:] != (self.num_state_tokens, self.dim):
            raise ValueError("state_tokens must have shape [B,num_state_tokens,dim]")
        batch = state_tokens.shape[0]
        state = self.state_projection(state_tokens.reshape(batch, self.num_state_tokens * self.dim))
        result = []
        for index, (history, mask) in enumerate(zip(histories, valid_masks)):
            if (history.ndim != 3 or history.shape[0] != batch or history.shape[-1] != self.dim or
                    mask.shape != history.shape[:2] or mask.dtype != torch.bool):
                raise ValueError("incompatible domain history/mask shapes or mask dtype")
            safe = torch.where(mask[..., None], history, 0)
            context = state[:, None].expand(-1, history.shape[1], -1)
            weights = torch.sigmoid(self.scorers[index](torch.cat((safe, context), -1)))
            weights = torch.where(mask[..., None], weights, 0)
            pooled = torch.einsum("blc,bld->bcd", weights, safe)
            # Accumulate in float32 for half types; zero vectors have no direction or gradient.
            norm_input = pooled.float() if pooled.dtype in (torch.float16, torch.bfloat16) else pooled
            normalized = F.normalize(norm_input, dim=-1, eps=1e-12)
            normalized = torch.where(norm_input.ne(0).any(-1, keepdim=True), normalized, 0)
            pooled = normalized.to(pooled.dtype)
            output = self.out_norm(self.pool_projections[index](pooled))
            result.append(torch.where(mask.any(-1)[:, None, None], output, 0))
        return torch.cat(result, 1)
