"""Small ranking and matching computations from named recommendation papers.

All inputs are already prepared tensors. Hosts own feature joins, negative sampling,
FM pretraining, and retrieval indexes. No external reference code is copied.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DeepCrossing(nn.Module):
    """Residual feature crossing; ``features [B,D] -> logits [B]``."""

    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int = 2):
        super().__init__()
        self.stack = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                   nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                                    for _ in range(num_blocks))
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.stack(features)
        for block in self.blocks:
            x = F.relu(x + block(x))
        return self.output(x).squeeze(-1)


class FNN(nn.Module):
    """FM-initializable field embeddings and MLP; IDs [B,F] -> logits [B].

    ``load_fm_embeddings`` loads pretrained per-field FM vectors; this class does
    not run FM pretraining. Without loading them it is an embedding/MLP baseline.
    """

    def __init__(self, field_cardinalities: tuple[int, ...], embedding_dim: int,
                 hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(n, embedding_dim)
                                        for n in field_cardinalities)
        sizes = (len(field_cardinalities) * embedding_dim, *hidden_dims, 1)
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def load_fm_embeddings(self, weights: list[torch.Tensor]) -> None:
        if len(weights) != len(self.embeddings):
            raise ValueError("one FM embedding matrix is required per field")
        for layer, weight in zip(self.embeddings, weights):
            if tuple(weight.shape) != tuple(layer.weight.shape):
                raise ValueError("FM embedding shape mismatch")
            with torch.no_grad():
                layer.weight.copy_(weight)

    def forward(self, field_ids: torch.Tensor) -> torch.Tensor:
        if field_ids.shape[-1] != len(self.embeddings):
            raise ValueError("field count mismatch")
        x = torch.cat([emb(field_ids[:, i]) for i, emb in enumerate(self.embeddings)], -1)
        return self.mlp(x).squeeze(-1)


class NeuMF(nn.Module):
    """Independent GMF and MLP ID embeddings; IDs [B] -> logits [B]."""

    def __init__(self, num_users: int, num_items: int, dim: int,
                 mlp_dims: tuple[int, ...] = (32, 16)):
        super().__init__()
        self.user_gmf = nn.Embedding(num_users, dim)
        self.item_gmf = nn.Embedding(num_items, dim)
        self.user_mlp = nn.Embedding(num_users, dim)
        self.item_mlp = nn.Embedding(num_items, dim)
        sizes = (2 * dim, *mlp_dims)
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers.extend((nn.Linear(a, b), nn.ReLU()))
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(dim + sizes[-1], 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gmf = self.user_gmf(user_ids) * self.item_gmf(item_ids)
        deep = self.mlp(torch.cat((self.user_mlp(user_ids), self.item_mlp(item_ids)), -1))
        return self.output(torch.cat((gmf, deep), -1)).squeeze(-1)


class LatentCross(nn.Module):
    """Context multiplication on hidden states [B,T,D] or [B,D]."""

    def __init__(self, hidden_dim: int, context_dim: int):
        super().__init__()
        self.context = nn.Linear(context_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gate = self.context(context)
        if hidden.ndim == 3 and gate.ndim == 2:
            gate = gate[:, None, :]
        return hidden * (1 + gate)


class MultiViewDSSM(nn.Module):
    """Shared user and domain-specific item encoders in a cosine space."""

    def __init__(self, user_dim: int, item_dims: tuple[int, ...], embedding_dim: int):
        super().__init__()
        self.user_tower = nn.Linear(user_dim, embedding_dim)
        self.item_towers = nn.ModuleList(nn.Linear(d, embedding_dim) for d in item_dims)

    def encode_user(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_tower(features), dim=-1)

    def encode_item(self, features: torch.Tensor, domain: int) -> torch.Tensor:
        return F.normalize(self.item_towers[domain](features), dim=-1)

    def forward(self, user: torch.Tensor, item: torch.Tensor, domain: int) -> torch.Tensor:
        return (self.encode_user(user) * self.encode_item(item, domain)).sum(-1)


class PredictiveBilinear(nn.Module):
    """Feature-based score ``user[B,U] @ W[U,I] · item[B,I]``."""

    def __init__(self, user_dim: int, item_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(user_dim, item_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bu,ui,bi->b", user, self.weight, item)


class DPPGreedy(nn.Module):
    """Fixed-capacity greedy determinant selection from a PSD kernel [B,N,N].

    Returns up to K indices [B,K], padded with -1 when no eligible conditional
    variance exceeds tolerance. Incremental Cholesky updates continue even when
    log-det gain is negative; this does not optimize unconstrained cardinality.
    Kernel construction and PSD validation belong to the caller.
    """

    def __init__(self, slate_size: int, tolerance: float = 1e-10):
        super().__init__()
        self.slate_size = slate_size
        self.tolerance = tolerance

    @torch.no_grad()
    def forward(self, kernel: torch.Tensor, eligibility: torch.Tensor) -> torch.Tensor:
        if kernel.shape[-1] != kernel.shape[-2] or eligibility.shape != kernel.shape[:2]:
            raise ValueError("kernel and eligibility shapes differ")
        batch, n, _ = kernel.shape
        result = torch.full((batch, self.slate_size), -1, dtype=torch.long,
                            device=kernel.device)
        for b in range(batch):
            matrix = kernel[b].float() if kernel.dtype in (torch.float16, torch.bfloat16) else kernel[b]
            c = matrix.new_zeros(n, self.slate_size)
            residual = matrix.diagonal().clone()
            available = eligibility[b].clone()
            for step in range(min(self.slate_size, n)):
                gains = residual.masked_fill(~available, float("-inf"))
                index = int(gains.argmax())
                if not torch.isfinite(gains[index]) or gains[index] <= self.tolerance:
                    break
                result[b, step] = index
                update = ((matrix[index] - (c[:, :step] * c[index, :step]).sum(-1)) /
                          gains[index].sqrt())
                c[:, step] = update
                residual = residual - update.square()
                available[index] = False
        return result


class RankNet(nn.Module):
    """Mean RankNet pair loss within each query row; ties/padding excluded."""

    def forward(self, scores: torch.Tensor, labels: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        scores = torch.where(mask, scores, torch.zeros_like(scores))
        labels = torch.where(mask, labels.to(scores.dtype), torch.zeros_like(scores))
        pairs = (labels[:, :, None] > labels[:, None, :]) & mask[:, :, None] & mask[:, None, :]
        loss = F.softplus(-(scores[:, :, None] - scores[:, None, :]))
        return (loss * pairs).sum() / pairs.sum().clamp_min(1)


class LambdaRank(nn.Module):
    """RankNet pair loss weighted by detached swap DeltaNDCG per query."""

    def forward(self, scores: torch.Tensor, labels: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        scores = torch.where(mask, scores, torch.zeros_like(scores))
        labels = torch.where(mask, labels.to(scores.dtype), torch.zeros_like(scores))
        n = scores.shape[1]
        pairs = (labels[:, :, None] > labels[:, None, :]) & mask[:, :, None] & mask[:, None, :]
        with torch.no_grad():
            order = scores.masked_fill(~mask, float("-inf")).argsort(-1, descending=True)
            position = torch.empty_like(order).scatter_(1, order,
                torch.arange(n, device=scores.device)[None].expand_as(order))
            discount = 1 / torch.log2(position.to(scores.dtype) + 2)
            gains = torch.pow(2.0, labels) - 1
            ideal = labels.masked_fill(~mask, float("-inf")).sort(-1, descending=True).values
            ideal_gain = torch.where(torch.isfinite(ideal), torch.pow(2.0, ideal) - 1, 0)
            normalizer = (ideal_gain / torch.log2(torch.arange(n, device=scores.device,
                             dtype=scores.dtype) + 2)).sum(-1).clamp_min(1e-12)
            delta = ((gains[:, :, None] - gains[:, None, :]) *
                     (discount[:, :, None] - discount[:, None, :])).abs() / normalizer[:, None, None]
        loss = F.softplus(-(scores[:, :, None] - scores[:, None, :]))
        weighted = delta * pairs
        return (loss * weighted).sum() / pairs.sum().clamp_min(1)
