"""Independent PyTorch cores for categorical feature interaction models.

Inputs are one integer ID per field, ``[batch, fields]``, except DCNv2, which
receives a preprocessed dense ``[batch, input_dim]`` vector. Outputs are logits.
"""

from itertools import combinations

import torch
from torch import nn


class _Fields(nn.Module):
    def __init__(self, cardinalities: list[int], embedding_dim: int):
        super().__init__()
        if len(cardinalities) < 2 or any(n <= 0 for n in cardinalities):
            raise ValueError("at least two positive field cardinalities are required")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.cardinalities = tuple(cardinalities)
        self.count = len(cardinalities)
        self.dim = embedding_dim
        self.pairs = tuple(combinations(range(self.count), 2))
        self.register_buffer("offsets", torch.tensor(
            [0] + [sum(cardinalities[:i]) for i in range(1, self.count)],
            dtype=torch.long))
        total = sum(cardinalities)
        self.embeddings = nn.Embedding(total, embedding_dim)
        self.linear = nn.Embedding(total, 1)
        self.bias = nn.Parameter(torch.zeros(()))

    def encode(self, fields: torch.Tensor) -> torch.Tensor:
        return self.embeddings(fields.long() + self.offsets)

    def first_order(self, fields: torch.Tensor) -> torch.Tensor:
        return self.linear(fields.long() + self.offsets).sum(dim=1).squeeze(-1) + self.bias

    def pair_products(self, vectors: torch.Tensor) -> torch.Tensor:
        return torch.stack([vectors[:, i] * vectors[:, j] for i, j in self.pairs], dim=1)


class AFM(nn.Module):
    """Attention-weighted pair products; categorical IDs ``[B,F]`` to logits ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 attention_dim: int = 16):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        self.attention = nn.Sequential(nn.Linear(embedding_dim, attention_dim),
                                       nn.ReLU(), nn.Linear(attention_dim, 1))
        self.head = nn.Linear(embedding_dim, 1, bias=False)

    def attention_weights(self, fields: torch.Tensor) -> torch.Tensor:
        pairs = self.fields.pair_products(self.fields.encode(fields))
        return torch.softmax(self.attention(pairs).squeeze(-1), dim=1)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        pairs = self.fields.pair_products(self.fields.encode(fields))
        weights = torch.softmax(self.attention(pairs).squeeze(-1), dim=1)
        return self.fields.first_order(fields) + self.head(
            (pairs * weights.unsqueeze(-1)).sum(dim=1)).squeeze(-1)


class AutoInt(nn.Module):
    """Field self-attention interactions; categorical IDs ``[B,F]`` to logits ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 num_heads: int = 2, num_layers: int = 2):
        super().__init__()
        if num_heads < 1 or embedding_dim % num_heads or num_layers < 1:
            raise ValueError("embedding_dim must divide num_heads and layers must be positive")
        self.fields = _Fields(cardinalities, embedding_dim)
        self.attentions = nn.ModuleList(nn.MultiheadAttention(
            embedding_dim, num_heads, batch_first=True) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(embedding_dim)
                                   for _ in range(num_layers))
        self.head = nn.Linear(self.fields.count * embedding_dim, 1)

    def interaction_features(self, fields: torch.Tensor) -> torch.Tensor:
        hidden = self.fields.encode(fields)
        for attention, norm in zip(self.attentions, self.norms):
            attended, _ = attention(hidden, hidden, hidden, need_weights=False)
            hidden = norm(hidden + attended)
        return hidden

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return self.fields.first_order(fields) + self.head(
            self.interaction_features(fields).flatten(1)).squeeze(-1)


class FiBiNET(nn.Module):
    """Field squeeze gates and bilinear pair terms; categorical IDs to ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 hidden_dim: int = 32):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        field_count = self.fields.count
        self.squeeze = nn.Sequential(nn.Linear(field_count, max(1, field_count // 2)),
                                     nn.ReLU(), nn.Linear(max(1, field_count // 2),
                                                          field_count), nn.Sigmoid())
        self.bilinear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.head = nn.Sequential(nn.Linear(2 * len(self.fields.pairs) * embedding_dim,
                                            hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def field_gates(self, fields: torch.Tensor) -> torch.Tensor:
        return self.squeeze(self.fields.encode(fields).mean(dim=-1))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        vectors = self.fields.encode(fields)
        gated = vectors * self.squeeze(vectors.mean(dim=-1)).unsqueeze(-1)
        products = [self.bilinear(vectors[:, i]) * vectors[:, j]
                    for i, j in self.fields.pairs]
        products += [self.bilinear(gated[:, i]) * gated[:, j]
                     for i, j in self.fields.pairs]
        return self.fields.first_order(fields) + self.head(
            torch.cat(products, dim=-1)).squeeze(-1)


class NFM(nn.Module):
    """Bi-interaction pooling followed by an MLP; categorical IDs to ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 hidden_dim: int = 32):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        self.deep = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, 1))

    @staticmethod
    def bi_interaction(vectors: torch.Tensor) -> torch.Tensor:
        return 0.5 * (vectors.sum(dim=1).square() - vectors.square().sum(dim=1))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return self.fields.first_order(fields) + self.deep(
            self.bi_interaction(self.fields.encode(fields))).squeeze(-1)


class PNN(nn.Module):
    """Inner and outer product layer before an MLP; categorical IDs to ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 hidden_dim: int = 32):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        pair_count = len(self.fields.pairs)
        input_dim = self.fields.count * embedding_dim + pair_count * (
            1 + embedding_dim * embedding_dim)
        self.deep = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, 1))

    def product_features(self, vectors: torch.Tensor) -> torch.Tensor:
        inner = torch.stack([(vectors[:, i] * vectors[:, j]).sum(dim=-1)
                             for i, j in self.fields.pairs], dim=1)
        outer = torch.cat([(vectors[:, i, :, None] * vectors[:, j, None, :]).flatten(1)
                           for i, j in self.fields.pairs], dim=1)
        return torch.cat([vectors.flatten(1), inner, outer], dim=1)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return self.fields.first_order(fields) + self.deep(
            self.product_features(self.fields.encode(fields))).squeeze(-1)


class WideDeep(nn.Module):
    """Explicit categorical crosses plus embedded MLP; IDs ``[B,F]`` to ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 hidden_dim: int = 32,
                 cross_pairs: list[tuple[int, int]] | None = None):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        self.cross_pairs = tuple(self.fields.pairs if cross_pairs is None else cross_pairs)
        if any(i < 0 or j >= self.fields.count or i >= j
               for i, j in self.cross_pairs):
            raise ValueError("cross_pairs must contain ordered field indices")
        self.cross_tables = nn.ModuleList(nn.Embedding(cardinalities[i] * cardinalities[j], 1)
                                          for i, j in self.cross_pairs)
        self.deep = nn.Sequential(nn.Flatten(),
                                  nn.Linear(self.fields.count * embedding_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, 1))

    def cross_indices(self, fields: torch.Tensor) -> torch.Tensor:
        if not self.cross_pairs:
            return fields.new_empty((fields.shape[0], 0))
        return torch.stack([fields[:, i] * self.fields.cardinalities[j] + fields[:, j]
                            for i, j in self.cross_pairs], dim=1)

    def wide_logit(self, fields: torch.Tensor) -> torch.Tensor:
        indices = self.cross_indices(fields)
        cross = sum((layer(indices[:, k]).squeeze(-1)
                     for k, layer in enumerate(self.cross_tables)),
                    torch.zeros(fields.shape[0], device=fields.device))
        return self.fields.first_order(fields) + cross

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return self.wide_logit(fields) + self.deep(self.fields.encode(fields)).squeeze(-1)


class XDeepFM(nn.Module):
    """Compressed interaction network plus MLP; categorical IDs to ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int,
                 hidden_dim: int = 32, cin_channels: list[int] | None = None):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        channels = [8, 8] if cin_channels is None else cin_channels
        if not channels or any(n <= 0 for n in channels):
            raise ValueError("cin_channels must contain positive sizes")
        self.cin_weights = nn.ParameterList()
        previous = self.fields.count
        for width in channels:
            weight = nn.Parameter(torch.empty(width, self.fields.count * previous))
            nn.init.xavier_uniform_(weight)
            self.cin_weights.append(weight)
            previous = width
        self.deep = nn.Sequential(nn.Flatten(),
                                  nn.Linear(self.fields.count * embedding_dim, hidden_dim),
                                  nn.ReLU())
        self.head = nn.Linear(sum(channels) + hidden_dim, 1)

    def cin_features(self, fields: torch.Tensor) -> torch.Tensor:
        original = self.fields.encode(fields)
        hidden = original
        outputs = []
        for weight in self.cin_weights:
            outer = (original[:, :, None, :] * hidden[:, None, :, :]).flatten(1, 2)
            hidden = torch.relu(torch.einsum("bmk,cm->bck", outer, weight))
            outputs.append(hidden.sum(dim=-1))
        return torch.cat(outputs, dim=1)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        vectors = self.fields.encode(fields)
        return self.fields.first_order(fields) + self.head(torch.cat(
            [self.cin_features(fields), self.deep(vectors)], dim=1)).squeeze(-1)


class DCNv2(nn.Module):
    """Matrix cross network with optional low rank plus MLP; dense ``[B,D]`` to ``[B]``."""

    def __init__(self, input_dim: int, num_cross_layers: int = 2,
                 hidden_dim: int = 32, rank: int | None = None):
        super().__init__()
        if input_dim <= 0 or num_cross_layers < 1 or (rank is not None and rank <= 0):
            raise ValueError("input_dim and cross layers must be positive; rank is positive")
        self.cross_layers = nn.ModuleList(
            nn.Sequential(nn.Linear(input_dim, rank, bias=False),
                          nn.Linear(rank, input_dim)) if rank is not None
            else nn.Linear(input_dim, input_dim)
            for _ in range(num_cross_layers))
        self.deep = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.head = nn.Linear(input_dim + hidden_dim, 1)

    def cross_features(self, x: torch.Tensor) -> torch.Tensor:
        crossed = x
        for layer in self.cross_layers:
            crossed = crossed + x * layer(crossed)
        return crossed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.cross_features(x), self.deep(x)], dim=1)).squeeze(-1)


class FwFM(nn.Module):
    """Field-pair-weighted FM; categorical IDs ``[B,F]`` to logits ``[B]``."""

    def __init__(self, cardinalities: list[int], embedding_dim: int):
        super().__init__()
        self.fields = _Fields(cardinalities, embedding_dim)
        self.pair_weights = nn.Parameter(torch.ones(len(self.fields.pairs)))

    def first_order(self, fields: torch.Tensor) -> torch.Tensor:
        return self.fields.first_order(fields)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        vectors = self.fields.encode(fields)
        pairs = torch.stack([(vectors[:, i] * vectors[:, j]).sum(dim=-1)
                             for i, j in self.fields.pairs], dim=1)
        return self.first_order(fields) + (pairs * self.pair_weights).sum(dim=1)
