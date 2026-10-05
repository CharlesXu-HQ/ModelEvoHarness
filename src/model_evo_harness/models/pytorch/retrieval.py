"""Independent PyTorch references for collaborative filtering and retrieval.

Inputs are already encoded tensors; host code owns positive-event filtering,
temporal splitting, negative sampling, and catalog indexing.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FunkSVD(nn.Module):
    """Pair score from user/item ID embeddings; forward(user_ids, item_ids) -> [B]."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return (self.user_embedding(user_ids) * self.item_embedding(item_ids)).sum(-1)


class BiasSVD(FunkSVD):
    """FunkSVD plus user, item, and global offsets; pair logits [B]."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__(num_users, num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return (super().forward(user_ids, item_ids) + self.user_bias(user_ids).squeeze(-1)
                + self.item_bias(item_ids).squeeze(-1) + self.global_bias)


class ItemCF:
    """Positive-only cosine item neighbors; fit([U,I]), scores([B,I]) -> [B,I]."""

    def fit(self, interactions: torch.Tensor) -> "ItemCF":
        positive = (interactions > 0).float()
        counts = positive.sum(0)
        scale = torch.sqrt(counts[:, None] * counts[None, :]).clamp_min(1)
        self.similarity = positive.T @ positive / scale
        self.similarity.fill_diagonal_(0)
        return self

    def scores(self, user_histories: torch.Tensor) -> torch.Tensor:
        return user_histories.float() @ self.similarity


class UserCF:
    """Positive-only cosine user neighbors; fit([U,I]), scores(user_ids) -> [B,I]."""

    def fit(self, interactions: torch.Tensor) -> "UserCF":
        self.interactions = (interactions > 0).float()
        counts = self.interactions.sum(1)
        scale = torch.sqrt(counts[:, None] * counts[None, :]).clamp_min(1)
        self.similarity = self.interactions @ self.interactions.T / scale
        self.similarity.fill_diagonal_(0)
        return self

    def scores(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.similarity[user_ids] @ self.interactions


class Swing:
    """Common-user-pair item similarity; fit([U,I]), scores([B,I]) -> [B,I].

    Each pair of users who interacted with *both* items contributes inverse
    activity and inverse shared-history weight. This dense reference is for
    small datasets; production use needs candidate-pair pruning.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, interactions: torch.Tensor) -> "Swing":
        positive = (interactions > 0).float()
        activity = positive.sum(1)
        shared = positive @ positive.T
        user_pair_weight = (activity[:, None] * activity[None, :]).clamp_min(1).rsqrt()
        user_pair_weight = user_pair_weight / (self.alpha + shared)
        user_pair_weight.fill_diagonal_(0)
        count = positive.shape[1]
        similarity = positive.new_zeros((count, count))
        for i in range(count):
            for j in range(i + 1, count):
                common = positive[:, i] * positive[:, j]
                score = 0.5 * common @ user_pair_weight @ common
                similarity[i, j] = score
                similarity[j, i] = score
        self.similarity = similarity
        return self

    def scores(self, user_histories: torch.Tensor) -> torch.Tensor:
        return user_histories.float() @ self.similarity


class EGES(nn.Module):
    """Item-specific mixture of ID/side embeddings scored against a context ID.

    forward(item_ids[B], side_ids[B,S], context_ids[B]) -> pair logits [B].
    """

    def __init__(self, num_items: int, side_cardinalities: list[int],
                 embedding_dim: int):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.side_embeddings = nn.ModuleList(nn.Embedding(n, embedding_dim)
                                             for n in side_cardinalities)
        self.attention = nn.Embedding(num_items, len(side_cardinalities) + 1)
        self.context_embedding = nn.Embedding(num_items, embedding_dim)

    def attention_weights(self, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.attention(item_ids), dim=-1)

    def encode_items(self, item_ids: torch.Tensor,
                     side_ids: torch.Tensor) -> torch.Tensor:
        sources = torch.stack([self.item_embedding(item_ids)] + [
            layer(side_ids[:, i]) for i, layer in enumerate(self.side_embeddings)], dim=1)
        return (sources * self.attention_weights(item_ids).unsqueeze(-1)).sum(1)

    def forward(self, item_ids: torch.Tensor, side_ids: torch.Tensor,
                context_ids: torch.Tensor) -> torch.Tensor:
        return (self.encode_items(item_ids, side_ids)
                * self.context_embedding(context_ids)).sum(-1)


class Item2Vec(nn.Module):
    """Skip-gram with negative sampling on ordered item-context pairs.

    negative_sampling_loss(center_ids[B], positive_ids[B], negative_ids[B,K]) -> scalar.
    user_vector(history_ids[B,T], mask[B,T]) -> mean input vector [B,D].
    """

    def __init__(self, num_items: int, embedding_dim: int):
        super().__init__()
        self.input_embedding = nn.Embedding(num_items, embedding_dim)
        self.output_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, center_ids: torch.Tensor,
                context_ids: torch.Tensor) -> torch.Tensor:
        return (self.input_embedding(center_ids)
                * self.output_embedding(context_ids)).sum(-1)

    def negative_sampling_loss(self, center_ids: torch.Tensor,
                               positive_ids: torch.Tensor,
                               negative_ids: torch.Tensor) -> torch.Tensor:
        center = self.input_embedding(center_ids)
        positive = self.forward(center_ids, positive_ids)
        negatives = (center[:, None, :] * self.output_embedding(negative_ids)).sum(-1)
        return -(F.logsigmoid(positive) + F.logsigmoid(-negatives).sum(1)).mean()

    def user_vector(self, history_ids: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        vectors = self.input_embedding(history_ids)
        active = mask.to(vectors.dtype)
        return (vectors * active[..., None]).sum(1) / active.sum(1, keepdim=True).clamp_min(1)


class DSSM(nn.Module):
    """Independent MLP towers with cosine pair score; forward([B,Du],[B,Di]) -> [B]."""

    def __init__(self, user_dim: int, item_dim: int, embedding_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.user_tower = nn.Sequential(nn.Linear(user_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, embedding_dim))
        self.item_tower = nn.Sequential(nn.Linear(item_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, embedding_dim))

    def encode_user(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_tower(features), dim=-1)

    def encode_item(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_tower(features), dim=-1)

    def forward(self, user_features: torch.Tensor,
                item_features: torch.Tensor) -> torch.Tensor:
        return (self.encode_user(user_features) * self.encode_item(item_features)).sum(-1)


class FMRecall(nn.Module):
    """Decomposed FM matching towers for one categorical ID per field.

    encode_user([B,U]), encode_item([B,I]) -> [B,D+1]; paired forward -> [B].
    The item tower carries item linear and item-item interaction terms.
    """

    def __init__(self, user_cardinalities: list[int], item_cardinalities: list[int],
                 embedding_dim: int):
        super().__init__()
        self.user_embeddings = nn.ModuleList(nn.Embedding(n, embedding_dim)
                                             for n in user_cardinalities)
        self.item_embeddings = nn.ModuleList(nn.Embedding(n, embedding_dim)
                                             for n in item_cardinalities)
        self.item_linear = nn.ModuleList(nn.Embedding(n, 1)
                                         for n in item_cardinalities)

    def encode_user(self, fields: torch.Tensor) -> torch.Tensor:
        vectors = torch.stack([layer(fields[:, i])
                               for i, layer in enumerate(self.user_embeddings)], dim=1)
        return torch.cat([vectors.new_ones((fields.shape[0], 1)), vectors.sum(1)], dim=1)

    def encode_item(self, fields: torch.Tensor) -> torch.Tensor:
        vectors = torch.stack([layer(fields[:, i])
                               for i, layer in enumerate(self.item_embeddings)], dim=1)
        summed = vectors.sum(1)
        pair_terms = 0.5 * (summed.square() - vectors.square().sum(1)).sum(1)
        linear = sum(layer(fields[:, i]).squeeze(-1)
                     for i, layer in enumerate(self.item_linear))
        return torch.cat([(linear + pair_terms).unsqueeze(1), summed], dim=1)

    def forward(self, user_fields: torch.Tensor,
                item_fields: torch.Tensor) -> torch.Tensor:
        return (self.encode_user(user_fields) * self.encode_item(item_fields)).sum(-1)


class YouTubeDNN(nn.Module):
    """User DNN and item-ID table for sampled-candidate retrieval.

    forward(user_features[B,Du], candidate_ids[K], history_ids[B,T] or None,
            history_mask[B,T] or None) -> scores[B,K]. Host chooses the sampled
    candidates and the cross-entropy target; this is not full sampled softmax.
    """

    def __init__(self, user_dim: int, num_items: int, embedding_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.history_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_tower = nn.Sequential(nn.Linear(user_dim + embedding_dim, hidden_dim),
                                        nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))
        self.embedding_dim = embedding_dim

    def encode_user(self, user_features: torch.Tensor,
                    history_ids: torch.Tensor | None = None,
                    history_mask: torch.Tensor | None = None) -> torch.Tensor:
        if history_ids is None:
            pooled = user_features.new_zeros((user_features.shape[0], self.embedding_dim))
        else:
            vectors = self.history_embedding(history_ids)
            active = (torch.ones_like(history_ids, dtype=vectors.dtype)
                      if history_mask is None else history_mask.to(vectors.dtype))
            pooled = (vectors * active[..., None]).sum(1) / active.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(self.user_tower(torch.cat([user_features, pooled], dim=1)),
                           dim=-1)

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_ids)

    def forward(self, user_features: torch.Tensor, candidate_ids: torch.Tensor,
                history_ids: torch.Tensor | None = None,
                history_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode_user(user_features, history_ids, history_mask) @ self.encode_item(candidate_ids).T


class YouTubeSBC(YouTubeDNN):
    """YouTubeDNN with log sampling-probability correction on the same candidates.

    forward(user_features, candidate_ids[K], sampling_probs[K], history_ids,
            history_mask) -> corrected scores[B,K]. The probabilities must
    describe this training sampler, not exposure or treatment assignment.
    """

    def forward(self, user_features: torch.Tensor, candidate_ids: torch.Tensor,
                sampling_probs: torch.Tensor,
                history_ids: torch.Tensor | None = None,
                history_mask: torch.Tensor | None = None) -> torch.Tensor:
        if torch.any(sampling_probs <= 0):
            raise ValueError("sampling_probs must be positive")
        return (super().forward(user_features, candidate_ids, history_ids, history_mask)
                - sampling_probs.log()[None, :])
