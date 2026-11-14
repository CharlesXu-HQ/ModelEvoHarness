"""Independent CTR auxiliary objectives. Host supplies aligned tensors and groups."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _weight(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("weights must be finite nonnegative numbers")
    return float(value)


def _labels(labels, batch, like):
    if labels.shape != (batch,) or labels.is_complex() or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must be binary [B]")
    return labels.detach().to(dtype=like.dtype, device=like.device)


def _finite(x):
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError("inputs must be finite real floating tensors")


def _mean(x, dim=None):
    # Divide before summing: finite large entries can overflow a sum-first mean.
    count = x.numel() if dim is None else x.shape[dim]
    return (x / count).sum(dim)


class EnsembleConsistency(nn.Module):
    """BCE plus sum of per-branch BCE to a detached ensemble, logits [B,M].

    logit_mean follows FinalNet's ensemble; probability_mean follows a stochastic
    prediction ensemble. Weight is a declared local adaptation; no external teacher.
    """
    def __init__(self, weight=1.0, aggregation="logit_mean"):
        super().__init__()
        self.weight = _weight(weight)
        if aggregation not in ("logit_mean", "probability_mean"):
            raise ValueError("aggregation must be logit_mean or probability_mean")
        self.aggregation = aggregation

    def forward(self, logits, labels):
        if logits.ndim != 2 or min(logits.shape) < 1:
            raise ValueError("logits must be nonempty [B,M]")
        _finite(logits)
        labels = _labels(labels, logits.shape[0], logits)
        if self.aggregation == "logit_mean":
            mean = _mean(logits, -1)
            supervised = _mean(F.binary_cross_entropy_with_logits(mean, labels, reduction="none"))
            teacher = torch.sigmoid(mean).detach()
        else:
            log_p = torch.logsumexp(F.logsigmoid(logits), -1) - math.log(logits.shape[1])
            log_q = torch.logsumexp(F.logsigmoid(-logits), -1) - math.log(logits.shape[1])
            supervised = -_mean(labels*log_p + (1-labels)*log_q)
            teacher = log_p.exp().detach()
        branch_losses = _mean(F.binary_cross_entropy_with_logits(
            logits, teacher[:, None].expand_as(logits), reduction="none"), 0)
        consistency = branch_losses.sum()
        return {"loss": supervised + (self.weight*branch_losses).sum(), "supervised": supervised,
                "consistency": consistency, "teacher": teacher}


class NextBehaviorObjective(nn.Module):
    """Aligned state/positive/negative transitions [B,T,*], True=valid mask [B,T].

    Host performs the next-event shift and supplies eligible negatives. Average
    positive and negative BCE over valid transitions; empty transitions yield zero.
    """
    def __init__(self, state_dim, event_dim, hidden_dim):
        super().__init__()
        if any(not isinstance(d, int) or isinstance(d, bool) or d < 1
               for d in (state_dim, event_dim, hidden_dim)):
            raise ValueError("dimensions must be positive integers")
        self.state_dim, self.event_dim = state_dim, event_dim
        self.scorer = nn.Sequential(nn.Linear(state_dim + event_dim, hidden_dim),
                                    nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, states, positive, negative, valid):
        if (states.ndim != 3 or states.shape[-1] != self.state_dim or states.shape[0] < 1 or
                positive.shape != (*states.shape[:2], self.event_dim) or
                negative.shape != positive.shape or valid.shape != states.shape[:2] or valid.dtype != torch.bool):
            raise ValueError("aligned transition shapes or boolean mask are invalid")
        states, positive, negative = [torch.where(valid[..., None], x, torch.zeros_like(x))
                                      for x in (states, positive, negative)]
        for x in (states, positive, negative):
            _finite(x)
        pos = self.scorer(torch.cat([states, positive], -1)).squeeze(-1)
        neg = self.scorer(torch.cat([states, negative], -1)).squeeze(-1)
        count = valid.sum()
        denominator = 2*count.clamp_min(1)
        terms = F.softplus(-pos) / denominator + F.softplus(neg) / denominator
        loss = torch.where(valid, terms, 0).sum()
        return {"loss": loss, "positive_logit": torch.where(valid, pos, 0),
                "negative_logit": torch.where(valid, neg, 0), "valid_count": count}


class CoarseFineObjective(nn.Module):
    """Weighted coarse/fine BCE on the same observed labels, raw scores [B]."""
    def __init__(self, coarse_weight=1.0, fine_weight=1.0):
        super().__init__()
        self.coarse_weight, self.fine_weight = _weight(coarse_weight), _weight(fine_weight)

    def forward(self, coarse_logit, fine_logit, labels):
        if coarse_logit.ndim != 1 or not coarse_logit.numel() or fine_logit.shape != coarse_logit.shape:
            raise ValueError("coarse/fine logits must have matching nonempty [B] shapes")
        _finite(coarse_logit)
        _finite(fine_logit)
        labels = _labels(labels, coarse_logit.shape[0], coarse_logit)
        coarse = _mean(F.binary_cross_entropy_with_logits(coarse_logit, labels, reduction="none"))
        fine = _mean(F.binary_cross_entropy_with_logits(fine_logit, labels, reduction="none"))
        return {"loss": self.coarse_weight*coarse + self.fine_weight*fine, "coarse": coarse, "fine": fine}


class GroupedRegularization(nn.Module):
    """Explicit lambda/p * sum(abs(parameter)**p), not decoupled weight decay.

    Host names nonoverlapping trainable parameter groups and excludes biases/norms
    when appropriate. Shared objects may appear only once, including within a group.
    """
    def __init__(self, embedding_weight=0.0, network_weight=0.0, p=2):
        super().__init__()
        self.embedding_weight, self.network_weight = _weight(embedding_weight), _weight(network_weight)
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 1:
            raise ValueError("p must be finite and >= 1")
        self.p = p

    def forward(self, embedding_parameters, network_parameters):
        groups = [list(embedding_parameters), list(network_parameters)]
        parameters = groups[0] + groups[1]
        if not parameters or len({id(x) for x in parameters}) != len(parameters):
            raise ValueError("parameter groups must be nonempty in total with no overlap or duplicate")
        for x in parameters:
            _finite(x)
        zero = (parameters[0]*0).sum()
        penalties = [sum(((x*0).sum() for x in group), zero) if weight == 0 else
                     weight/self.p * sum((x.abs().pow(self.p).sum() for x in group), zero)
                     for group, weight in zip(groups, (self.embedding_weight, self.network_weight))]
        return {"loss": sum(penalties), "embedding": penalties[0], "network": penalties[1]}
