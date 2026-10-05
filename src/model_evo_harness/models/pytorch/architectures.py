"""Small, independently written PyTorch recommendation model references."""

import torch
from torch import nn


class FM(nn.Module):
    """Second-order factorization machine for one categorical ID per field."""

    def __init__(self, cardinalities: list[int], embedding_dim: int):
        super().__init__()
        self.linear = nn.ModuleList(nn.Embedding(n, 1) for n in cardinalities)
        self.embeddings = nn.ModuleList(nn.Embedding(n, embedding_dim) for n in cardinalities)
        self.bias = nn.Parameter(torch.zeros(()))

    def _parts(self, fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        linear = sum(layer(fields[:, i]).squeeze(-1)
                     for i, layer in enumerate(self.linear)) + self.bias
        vectors = torch.stack([layer(fields[:, i])
                               for i, layer in enumerate(self.embeddings)], dim=1)
        summed = vectors.sum(dim=1)
        interaction = 0.5 * (summed.square() - vectors.square().sum(dim=1)).sum(dim=1)
        return linear + interaction, vectors

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return self._parts(fields)[0]


class DeepFM(FM):
    """FM and MLP trained jointly from the same field embeddings."""

    def __init__(self, cardinalities: list[int], embedding_dim: int, hidden_dim: int):
        super().__init__(cardinalities, embedding_dim)
        self.deep = nn.Sequential(nn.Flatten(),
                                  nn.Linear(len(cardinalities) * embedding_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        fm_logit, vectors = self._parts(fields)
        return fm_logit + self.deep(vectors).squeeze(-1)


class DCN(nn.Module):
    """Original vector-cross DCN with a parallel MLP branch."""

    def __init__(self, input_dim: int, num_cross_layers: int, hidden_dim: int):
        super().__init__()
        self.cross_weights = nn.ModuleList(nn.Linear(input_dim, 1, bias=False)
                                           for _ in range(num_cross_layers))
        self.cross_biases = nn.ParameterList(nn.Parameter(torch.zeros(input_dim))
                                             for _ in range(num_cross_layers))
        self.deep = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.head = nn.Linear(input_dim + hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        crossed = x
        for weight, bias in zip(self.cross_weights, self.cross_biases):
            crossed = crossed + x * weight(crossed) + bias
        return self.head(torch.cat([crossed, self.deep(x)], dim=-1)).squeeze(-1)


class DIN(nn.Module):
    """Target-conditioned attention over a supplied history embedding sequence."""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(4 * embedding_dim, hidden_dim),
                                       nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(nn.Linear(2 * embedding_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, target: torch.Tensor, history: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        repeated = target.unsqueeze(1).expand_as(history)
        attention_input = torch.cat([repeated, history, repeated - history,
                                     repeated * history], dim=-1)
        scores = self.attention(attention_input).squeeze(-1)
        # DIN retains activation intensity instead of normalizing weights to sum to one.
        weights = torch.sigmoid(scores) * mask.to(scores.dtype)
        pooled = (history * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(torch.cat([target, pooled], dim=-1)).squeeze(-1)


class MMoE(nn.Module):
    """Multiple shared experts with an independent gate and head per task."""

    def __init__(self, input_dim: int, num_tasks: int, num_experts: int,
                 expert_dim: int):
        super().__init__()
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(input_dim, expert_dim),
                                                   nn.ReLU()) for _ in range(num_experts))
        self.gates = nn.ModuleList(nn.Linear(input_dim, num_experts)
                                   for _ in range(num_tasks))
        self.heads = nn.ModuleList(nn.Linear(expert_dim, 1) for _ in range(num_tasks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_values = torch.stack([expert(x) for expert in self.experts], dim=1)
        outputs = []
        for gate, head in zip(self.gates, self.heads):
            weights = torch.softmax(gate(x), dim=1).unsqueeze(-1)
            outputs.append(head((expert_values * weights).sum(dim=1)))
        return torch.cat(outputs, dim=1)


class TwoTower(nn.Module):
    """Independent user/item encoders with a batchwise dot-product score matrix."""

    def __init__(self, user_dim: int, item_dim: int, embedding_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.user_tower = nn.Sequential(nn.Linear(user_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, embedding_dim))
        self.item_tower = nn.Sequential(nn.Linear(item_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, embedding_dim))

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_features)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_features)

    def forward(self, user_features: torch.Tensor,
                item_features: torch.Tensor) -> torch.Tensor:
        return self.encode_user(user_features) @ self.encode_item(item_features).T
