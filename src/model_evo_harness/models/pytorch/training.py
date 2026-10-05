"""Small, framework-native objectives and sampling operations for experiments.

The host owns labels, exposure eligibility, train/validation splits, and metric
selection. These functions only transform tensors supplied by that host.
"""

import torch
from torch.nn import functional as F


def focal_loss(logits, labels, gamma=2.0, alpha=0.25):
    """Binary focal loss; ``alpha`` weights positives, ``1-alpha`` negatives."""
    labels = labels.to(dtype=logits.dtype)
    ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p_t = torch.exp(-ce)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).mean()


def bpr_loss(positive_scores, negative_scores):
    """Pairwise Bayesian personalized ranking objective."""
    return F.softplus(negative_scores - positive_scores).mean()


def listwise_loss(scores, relevance, candidate_mask):
    """Cross entropy between observed relevance and scores within each slate."""
    masked_scores = scores.masked_fill(~candidate_mask.bool(), torch.finfo(scores.dtype).min)
    masked_relevance = relevance.to(scores.dtype) * candidate_mask.to(scores.dtype)
    target = masked_relevance / masked_relevance.sum(dim=-1, keepdim=True).clamp_min(1)
    valid = masked_relevance.sum(dim=-1) > 0
    if not bool(valid.any()):
        raise ValueError("listwise_loss needs at least one slate with observed relevance")
    per_slate = -(target * F.log_softmax(masked_scores, dim=-1)).sum(dim=-1)
    return per_slate[valid].mean()


def hard_negative_indices(queries, items, allowed_negative_mask, k):
    """Return the hardest *eligible* item IDs for each query by dot product.

    The caller must exclude positives, unexposed items, and validation rows in
    ``allowed_negative_mask`` before calling this function.
    """
    if k < 1 or not bool((allowed_negative_mask.sum(dim=-1) >= k).all()):
        raise ValueError("each query needs at least k eligible negative candidates")
    scores = queries @ items.T
    scores = scores.masked_fill(~allowed_negative_mask.bool(), torch.finfo(scores.dtype).min)
    return scores.topk(k, dim=-1).indices
