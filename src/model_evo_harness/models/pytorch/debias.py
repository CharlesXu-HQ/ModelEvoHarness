"""Cold-start and bias tensor cores. Inputs, sampling and training belong to the host.

PAL and MetaEmbedding are independent paper-derived implementations. DICE's
masked objectives follow the MIT reference by Zheng et al.; see THIRD_PARTY_NOTICES.md.
"""

# DICE masked objective adapted from FIB LAB, Tsinghua University
# https://github.com/tsinghua-fib-lab/DICE/blob/591b883cd62bf0d88c906d12ed2ab88ff8fccc36/src/model.py
# MIT License
# 
# Copyright (c) 2022 FIB LAB, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch import nn
from torch.nn import functional as F


class PAL(nn.Module):
    """Position-only examination × relevance probability; outputs three [B] tensors.

    Pass position-free relevance logits from the host backbone. Call relevance()
    for serving; observed click probability is only for the logged-position loss.
    Factorization is an assumption, not an identification guarantee.
    """

    def __init__(self, num_positions: int):
        super().__init__()
        self.examination = nn.Embedding(num_positions, 1)
        nn.init.zeros_(self.examination.weight)

    def forward(self, relevance_logits, position_ids):
        exposure = torch.sigmoid(self.examination(position_ids).squeeze(-1))
        relevance = self.relevance(relevance_logits)
        return {"relevance": relevance, "examination": exposure,
                "observed": relevance * exposure}

    def relevance(self, relevance_logits):
        return torch.sigmoid(relevance_logits)

    def objective(self, relevance_logits, position_ids, labels):
        """Mean joint BCE, evaluated stably without clipping probability products."""
        a, b = relevance_logits, self.examination(position_ids).squeeze(-1)
        joint_logit = a + b - torch.logsumexp(torch.stack((torch.zeros_like(a), a, b)), 0)
        return F.binary_cross_entropy_with_logits(joint_logit, labels)


class MetaEmbedding(nn.Module):
    """Attribute-conditioned ID initialization [B,A] -> [B,D], not a meta-trainer.

    Attributes are detached frozen representations. The host supplies paired
    support/query examples for each entity and differentiates through adapt();
    ordinary supervised fitting of this generator is not meta-learning.
    """

    def __init__(self, attribute_dim: int, embedding_dim: int):
        super().__init__()
        self.generator = nn.Linear(attribute_dim, embedding_dim, bias=False)

    def forward(self, attributes):
        return torch.tanh(self.generator(attributes.detach()))

    def adapt(self, embedding, support_gradient, step_size: float):
        """Functional inner update; use create_graph=True for the support gradient."""
        return embedding - step_size * support_gradient

    def meta_loss(self, cold_loss, warm_loss, alpha: float = 0.1):
        return alpha * cold_loss + (1 - alpha) * warm_loss


class DICE(nn.Module):
    """Separate interest/conformity factors; pair ID tensors [B] -> three [B] scores.

    objective() requires mask=True iff the negative is more popular than the
    positive under a host-defined popularity protocol. It does not construct
    causal samples or run curriculum training. Negative L2 discrepancy promotes
    separation but is unbounded without host regularization; default weight is 0.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_interest = nn.Embedding(num_users, embedding_dim)
        self.user_conformity = nn.Embedding(num_users, embedding_dim)
        self.item_interest = nn.Embedding(num_items, embedding_dim)
        self.item_conformity = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        interest = (self.user_interest(user_ids) * self.item_interest(item_ids)).sum(-1)
        conformity = (self.user_conformity(user_ids) * self.item_conformity(item_ids)).sum(-1)
        return {"interest": interest, "conformity": conformity, "total": interest + conformity}

    def objective(self, user_ids, positive_ids, negative_ids, negative_more_popular,
                  interest_weight: float = 0.1, conformity_weight: float = 0.1,
                  discrepancy_weight: float = 0.0):
        p, n = self(user_ids, positive_ids), self(user_ids, negative_ids)
        mask = negative_more_popular.to(p["total"].dtype)
        interest = (mask * F.softplus(n["interest"] - p["interest"])).mean()
        conformity = (mask * F.softplus(p["conformity"] - n["conformity"])
                      + (1 - mask) * F.softplus(n["conformity"] - p["conformity"])).mean()
        users = torch.unique(user_ids)
        items = torch.unique(torch.cat((positive_ids, negative_ids)))
        discrepancy = F.mse_loss(self.user_interest(users), self.user_conformity(users))
        discrepancy = discrepancy + F.mse_loss(self.item_interest(items), self.item_conformity(items))
        return (F.softplus(n["total"] - p["total"]).mean() + interest_weight * interest
                + conformity_weight * conformity - discrepancy_weight * discrepancy)
