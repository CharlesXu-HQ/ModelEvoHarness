"""RankMixer dense block and UniDot cross-layer interaction readout.

RankMixer is independently implemented from arXiv:2507.15551v3, Eq. 1, 3-8.
Highway adapts FMHighway and per-domain dots from linrongc/kdd26_unidot,
commit 1399702006a1fc2f86329c8c6133f3f3ac1e6d0d, unified_model.py.
Changes: stacked tensor API, optional domain bundle, explicit validation,
no classifier or bus dependencies. The paper's aggregated Gram is omitted.

MIT License (UniDot portions)
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


class RankMixerBlock(nn.Module):
    """[B,T,D] -> [B,T,D]; fixed, fully defined field tokens, D divisible by T.

    LN(x + rewire(x)), then LN(s + independent_per_token_GELU_FFN(s)).
    No tokenizer, head, sparse experts or sequence handling.
    """

    def __init__(self, num_tokens: int, dim: int, hidden_dim: int):
        super().__init__()
        if min(num_tokens, dim, hidden_dim) < 1:
            raise ValueError("dimensions must be positive")
        if dim % num_tokens:
            raise ValueError("dim must be divisible by num_tokens")
        self.num_tokens, self.dim = num_tokens, dim
        self.w1 = nn.Parameter(torch.empty(num_tokens, dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_tokens, hidden_dim, dim))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, dim))
        for token in range(num_tokens):
            nn.init.xavier_uniform_(self.w1[token])
            nn.init.xavier_uniform_(self.w2[token])
        self.mix_norm = nn.LayerNorm(dim, eps=1e-5)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-5)

    def mix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Exchange token/head axes; this permutation alone is self-inverse."""
        if tokens.ndim != 3 or tokens.shape[1:] != (self.num_tokens, self.dim):
            raise ValueError("tokens must have shape [B,num_tokens,dim]")
        return tokens.reshape(-1, self.num_tokens, self.num_tokens,
                              self.dim // self.num_tokens).transpose(1, 2).reshape_as(tokens)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        states = self.mix_norm(tokens + self.mix_tokens(tokens))
        hidden = F.gelu(torch.einsum("btd,tdh->bth", states, self.w1) + self.b1)
        delta = torch.einsum("bth,thd->btd", hidden, self.w2) + self.b2
        return self.ffn_norm(states + delta)


class ExplicitInteractionHighway(nn.Module):
    """Read out all K layers into [B,K*S*I + K*U*I], domains first.

    user_states [B,K,U,D] and item_states [B,K,I,D] are post-fusion states.
    Optional domain_dots [B,K,S,I] come from pre-residual queries/readouts,
    not from these post-fusion item states. LayerNorm is shared across layers
    on each side; flattened domain and cross-bus bundles are normalized
    separately. Feed the returned vector into a host-owned prediction head.
    """

    def __init__(self, num_layers: int, dim: int, num_user_tokens: int,
                 num_item_tokens: int, num_domains: int = 0):
        super().__init__()
        if min(num_layers, dim, num_user_tokens, num_item_tokens) < 1 or num_domains < 0:
            raise ValueError("dimensions must be positive; num_domains may be zero")
        self.user_shape = (num_layers, num_user_tokens, dim)
        self.item_shape = (num_layers, num_item_tokens, dim)
        self.domain_shape = (num_layers, num_domains, num_item_tokens)
        self.cross_width = num_layers * num_user_tokens * num_item_tokens
        self.domain_width = num_layers * num_domains * num_item_tokens
        if self.cross_width < 2 or self.domain_width == 1:
            raise ValueError("each normalized interaction bundle needs at least two values")
        self.user_norm = nn.LayerNorm(dim, eps=1e-5)
        self.item_norm = nn.LayerNorm(dim, eps=1e-5)
        self.cross_norm = nn.LayerNorm(self.cross_width, eps=1e-5)
        self.domain_norm = nn.LayerNorm(self.domain_width, eps=1e-5) if self.domain_width else None

    @staticmethod
    def domain_interactions(queries: torch.Tensor, readouts: torch.Tensor,
                            valid_domains: torch.Tensor) -> torch.Tensor:
        """Raw aligned dots: [B,I,D], [B,S,I,D], bool [B,S] -> [B,S,I].

        The pinned author code uses identity normalization here. True means
        an available domain. Masking is before arithmetic; no attention or
        cross-domain retrieval is performed. Final bundle LN can recenter zeros.
        """
        if (queries.ndim != 3 or readouts.ndim != 4 or
                readouts.shape[0] != queries.shape[0] or readouts.shape[2:] != queries.shape[1:] or
                valid_domains.shape != readouts.shape[:2] or valid_domains.dtype != torch.bool):
            raise ValueError("incompatible query/readout/domain mask shapes or mask dtype")
        safe = torch.where(valid_domains[..., None, None], readouts, 0)
        return (queries[:, None] * safe).sum(-1)

    def forward(self, user_states: torch.Tensor, item_states: torch.Tensor,
                domain_dots: torch.Tensor | None = None) -> torch.Tensor:
        if (user_states.ndim != 4 or item_states.ndim != 4 or
                user_states.shape[1:] != self.user_shape or item_states.shape[1:] != self.item_shape or
                user_states.shape[0] != item_states.shape[0]):
            raise ValueError("user/item states must have the configured [B,K,T,D] shapes")
        batch = user_states.shape[0]
        cross = torch.matmul(self.user_norm(user_states), self.item_norm(item_states).transpose(-2, -1))
        cross = self.cross_norm(cross.reshape(batch, self.cross_width))
        if self.domain_width:
            if domain_dots is None or domain_dots.shape != (batch, *self.domain_shape):
                raise ValueError("domain_dots must have the configured [B,K,S,I] shape")
            domains = self.domain_norm(domain_dots.reshape(batch, self.domain_width))
            return torch.cat((domains, cross), -1)
        if domain_dots is not None:
            raise ValueError("domain_dots require num_domains > 0")
        return cross
