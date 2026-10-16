"""Small paper-derived graph tensor cores; callers prepare graphs and samples.

These modules do not reproduce graph sampling, training schedules, or serving systems.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _attention(logits, mask):
    active = mask.bool()
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    masked = work.masked_fill(~active, float("-inf"))
    masked = torch.where(active.any(-1, keepdim=True), masked, torch.zeros_like(masked))
    return torch.where(active, torch.softmax(masked, -1), 0).to(logits.dtype)


class GraphSAGE(nn.Module):
    """Mean GraphSAGE layer: forward(self[B,D], neighbors[B,K,D], mask[B,K]) -> [B,O].

    Caller supplies sampled neighbors. This is one aggregation layer, not the
    unsupervised random-walk objective from the GraphSAGE paper.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(2 * input_dim, output_dim)

    def forward(self, self_features, neighbor_features, neighbor_mask):
        active = neighbor_mask.to(neighbor_features.dtype)
        mean = (neighbor_features * active[..., None]).sum(1) / active.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(F.relu(self.projection(torch.cat((self_features, mean), -1))), dim=-1)


class PinSage(nn.Module):
    """One PinSage convolution; caller provides random-walk visit weights.

    forward(self[B,D], neighbors[B,K,D], importance[B,K], mask[B,K]) -> [B,O].
    No walk generation, top-T selection, curriculum negatives, or MapReduce.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.neighbor_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(input_dim + hidden_dim, output_dim)

    def forward(self, self_features, neighbor_features, importance, neighbor_mask):
        weights = importance * neighbor_mask.to(importance.dtype)
        transformed = F.relu(self.neighbor_projection(neighbor_features))
        pooled = (transformed * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-12)
        return F.normalize(F.relu(self.output_projection(torch.cat((self_features, pooled), -1))), dim=-1)


class SDNE(nn.Module):
    """SDNE autoencoder and first/second-order losses on caller-supplied adjacency.

    forward(adjacency_rows[B,N]) -> (embeddings[B,D], reconstruction[B,N]).
    training_loss(adjacency[N,N], edge_index[E,2], edge_weight[E]) -> scalar; edge
    indices address the N rows. Dense adjacency makes this a small-graph core.
    """

    def __init__(self, num_nodes: int, hidden_dim: int, embedding_dim: int,
                 beta: float = 5.0, alpha: float = 1.0, regularization: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(num_nodes, hidden_dim), nn.Sigmoid(),
                                     nn.Linear(hidden_dim, embedding_dim), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.Sigmoid(),
                                     nn.Linear(hidden_dim, num_nodes), nn.Sigmoid())
        self.beta, self.alpha, self.regularization = beta, alpha, regularization

    def forward(self, adjacency_rows):
        embedding = self.encoder(adjacency_rows)
        return embedding, self.decoder(embedding)

    def training_loss(self, adjacency, edge_index, edge_weight):
        embedding, reconstruction = self(adjacency)
        penalty = torch.where(adjacency != 0, self.beta, 1.0)
        second_order = ((reconstruction - adjacency) * penalty).square().sum()
        first_order = (edge_weight * (embedding[edge_index[:, 0]]
                                      - embedding[edge_index[:, 1]]).square().sum(-1)).sum()
        regularizer = sum(parameter.square().sum() for parameter in self.parameters())
        return second_order + self.alpha * first_order + self.regularization * regularizer


class LINE(nn.Module):
    """First- or second-order LINE with caller-provided weighted edges/negatives.

    training_loss(source[B], target[B], negative_targets[B,K], edge_weight[B]) -> scalar.
    First order shares endpoint embeddings and requires undirected edge semantics;
    second order uses separate source/context embeddings for directed edges.
    Sampling, edge alias tables, and concatenating separately trained orders
    are outside this tensor core.
    """

    def __init__(self, num_nodes: int, embedding_dim: int, order: int = 2):
        super().__init__()
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.order = order
        self.source_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.context_embedding = (self.source_embedding if order == 1
                                  else nn.Embedding(num_nodes, embedding_dim))

    def forward(self, source_ids, target_ids):
        return (self.source_embedding(source_ids) * self.context_embedding(target_ids)).sum(-1)

    def training_loss(self, source_ids, target_ids, negative_target_ids, edge_weight):
        source = self.source_embedding(source_ids)
        positive = self.forward(source_ids, target_ids)
        negative = (source[:, None, :] * self.context_embedding(negative_target_ids)).sum(-1)
        per_edge = F.softplus(-positive) + F.softplus(negative).sum(-1)
        return (per_edge * edge_weight).sum() / edge_weight.sum().clamp_min(torch.finfo(edge_weight.dtype).tiny)


class RippleNet(nn.Module):
    """Multi-hop KG response for candidate items with preassembled ripple triples.

    forward(item_ids[B], heads[H,B,M], relations[H,B,M], tails[H,B,M],
    mask[H,B,M]) -> click probabilities[B]. item_update is one of none,
    plus, replace; every hop's response contributes to the user vector.
    This core omits ripple-set construction and the paper's KG auxiliary loss.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 num_hops: int, item_update: str = "none"):
        super().__init__()
        if item_update not in ("none", "plus", "replace"):
            raise ValueError("item_update must be none, plus, or replace")
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim * embedding_dim)
        self.embedding_dim, self.num_hops, self.item_update = embedding_dim, num_hops, item_update

    def forward(self, item_ids, heads, relations, tails, mask):
        item = self.entity_embedding(item_ids)
        responses = []
        for hop in range(self.num_hops):
            head = self.entity_embedding(heads[hop])
            tail = self.entity_embedding(tails[hop])
            relation = self.relation_embedding(relations[hop]).reshape(*head.shape[:-1],
                                                                        self.embedding_dim, self.embedding_dim)
            logits = torch.einsum("bi,bmij,bmj->bm", item, relation, head)
            active = mask[hop].bool()
            attention = _attention(logits, active)
            response = (attention[..., None] * tail).sum(1)
            responses.append(response)
            if self.item_update == "plus":
                item = item + response
            elif self.item_update == "replace":
                item = response
        user = torch.stack(responses, dim=0).sum(0)
        return torch.sigmoid((user * item).sum(-1))


class KGAT(nn.Module):
    """Single relation-aware attention and bi-interaction propagation layer.

    forward(head_ids[B], neighbor_ids[B,K], relation_ids[B,K], mask[B,K])
    -> [B,O]. Caller builds the collaborative KG and composes user/item
    propagated vectors into a score. The
    TransR KG loss, BPR loss, attention-refresh schedule, and deeper layers
    are not included.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 output_dim: int):
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.relation_projection = nn.Embedding(num_relations, embedding_dim * embedding_dim)
        self.sum_projection = nn.Linear(embedding_dim, output_dim, bias=False)
        self.product_projection = nn.Linear(embedding_dim, output_dim, bias=False)
        self.embedding_dim = embedding_dim

    def forward(self, head_ids, neighbor_ids, relation_ids, neighbor_mask):
        head = self.entity_embedding(head_ids)
        tail = self.entity_embedding(neighbor_ids)
        relation = self.relation_embedding(relation_ids)
        projection = self.relation_projection(relation_ids).reshape(*tail.shape[:-1],
                                                                     self.embedding_dim, self.embedding_dim)
        projected_tail = torch.einsum("bkij,bkj->bki", projection, tail)
        projected_head = torch.einsum("bkij,bj->bki", projection, head)
        logits = (projected_tail * torch.tanh(projected_head + relation)).sum(-1)
        active = neighbor_mask.bool()
        attention = _attention(logits, active)
        neighborhood = (attention[..., None] * tail).sum(1)
        return (F.leaky_relu(self.sum_projection(head + neighborhood), negative_slope=0.2)
                + F.leaky_relu(self.product_projection(head * neighborhood), negative_slope=0.2))


class CDL(nn.Module):
    """Deterministic MAP-style CDL content autoencoder coupled to latent MF.

    forward(user_ids[B], item_ids[B], corrupted_content[B,C]) ->
    (rating[B], reconstruction[B,C], content_code[B,D]). training_loss additionally
    takes clean_content[B,C], rating[B], confidence[B]. Caller corrupts input
    and supplies confidence weights. This paired minibatch objective repeats
    content/coupling terms for repeated items; it is not the Bayesian sampler.
    """

    def __init__(self, num_users: int, num_items: int, content_dim: int,
                 hidden_dim: int, embedding_dim: int, reconstruction_weight: float = 1.0,
                 coupling_weight: float = 1.0, regularization: float = 0.0):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.encoder = nn.Sequential(nn.Linear(content_dim, hidden_dim), nn.Sigmoid(),
                                     nn.Linear(hidden_dim, embedding_dim), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.Sigmoid(),
                                     nn.Linear(hidden_dim, content_dim), nn.Sigmoid())
        self.reconstruction_weight = reconstruction_weight
        self.coupling_weight = coupling_weight
        self.regularization = regularization

    def forward(self, user_ids, item_ids, corrupted_content):
        code = self.encoder(corrupted_content)
        reconstruction = self.decoder(code)
        rating = (self.user_embedding(user_ids) * self.item_embedding(item_ids)).sum(-1)
        return rating, reconstruction, code

    def training_loss(self, user_ids, item_ids, corrupted_content, clean_content, rating, confidence):
        prediction, reconstruction, code = self(user_ids, item_ids, corrupted_content)
        feedback = (confidence * (prediction - rating).square()).mean()
        content = (reconstruction - clean_content).square().sum(-1).mean()
        coupling = (self.item_embedding(item_ids) - code).square().sum(-1).mean()
        regularizer = (self.user_embedding(user_ids).square().sum(-1).mean()
                       + self.item_embedding(item_ids).square().sum(-1).mean())
        return (feedback + self.reconstruction_weight * content
                + self.coupling_weight * coupling + self.regularization * regularizer)
