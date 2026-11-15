"""Native target-conditioned pooling and padding-aware Dice activation.

Dice is an activation, distinct from interest/conformity disentanglement DICE.
These layers use pre-decision embeddings; no external recommendation framework.
"""
import math

import torch
from torch import nn


def _clean(values, valid):
    if valid.dtype != torch.bool or valid.shape != values.shape[:-1]:
        raise ValueError("boolean mask must match all input axes except feature width")
    values = torch.where(valid[..., None], values, 0.)
    if not values.is_floating_point() or not torch.isfinite(values).all().item():
        raise ValueError("valid inputs must be finite floating values")
    return values


class MaskedDice(nn.Module):
    """Distribution gate with learned alpha; padding excluded from statistics.

    Moments use all valid positions across leading axes, population variance,
    and moving = momentum * moving + (1-momentum) * batch. Empty training
    batches do not update moments. Evaluation uses frozen moving moments.
    """
    def __init__(self, dim, momentum=.9, epsilon=1e-5):
        super().__init__()
        if dim < 1 or not 0 <= momentum < 1 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("positive dim/epsilon and momentum in [0,1) required")
        self.dim, self.momentum, self.epsilon = dim, momentum, epsilon
        self.alpha = nn.Parameter(torch.zeros(dim))
        self.register_buffer('moving_mean', torch.zeros(dim))
        self.register_buffer('moving_variance', torch.ones(dim))

    def forward(self, values, valid):
        values = _clean(values, valid)
        if values.ndim < 2 or values.shape[-1] != self.dim:
            raise ValueError("expected [...,dim] values")
        mean, variance = self.moving_mean, self.moving_variance
        if self.training and valid.any().item():
            selected = values[valid]
            mean = selected.mean(0)
            variance = (selected - mean).square().mean(0)
            with torch.no_grad():
                self.moving_mean.mul_(self.momentum).add_(mean.detach(), alpha=1-self.momentum)
                self.moving_variance.mul_(self.momentum).add_(variance.detach(), alpha=1-self.momentum)
        gate = torch.sigmoid((values - mean) * torch.rsqrt(variance + self.epsilon))
        output = values * (gate + self.alpha * (1-gate))
        return torch.where(valid[..., None], output, 0.)


class LocalActivationPooling(nn.Module):
    """MLP([target,event,target-event,target*event]) then masked pooling.

    raw retains signed activation intensity; sigmoid bounds each independent
    weight; softmax normalizes across valid events. Empty readouts are zero.
    Returns representation [B,D], attention and masked scores [B,T].
    """
    def __init__(self, dim, hidden_dim, activation='relu', normalization='sigmoid'):
        super().__init__()
        if min(dim, hidden_dim) < 1 or activation not in ('relu', 'dice'):
            raise ValueError("positive sizes and relu/dice activation required")
        if normalization not in ('raw', 'sigmoid', 'softmax'):
            raise ValueError("normalization must be raw, sigmoid or softmax")
        self.dim, self.normalization = dim, normalization
        self.hidden = nn.Linear(4*dim, hidden_dim)
        self.activation = MaskedDice(hidden_dim) if activation == 'dice' else nn.ReLU()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, target, history, valid):
        if (history.ndim != 3 or target.shape != (history.shape[0], self.dim)
                or history.shape[-1] != self.dim or not torch.isfinite(target).all().item()):
            raise ValueError("finite target [B,D] and history [B,T,D] required")
        history = _clean(history, valid)
        query = target[:, None].expand_as(history)
        features = torch.cat([query, history, query-history, query*history], -1)
        hidden = self.hidden(features)
        hidden = self.activation(hidden, valid) if isinstance(self.activation, MaskedDice) else self.activation(hidden)
        scores = self.score(hidden).squeeze(-1)
        if self.normalization == 'softmax':
            masked = torch.where(valid, scores, float('-inf'))
            dummy = torch.where(valid.any(-1, keepdim=True), float('-inf'), 0.)
            weights = torch.softmax(torch.cat([masked, dummy], -1), -1)[:, :-1]
        else:
            weights = scores if self.normalization == 'raw' else scores.sigmoid()
        weights = torch.where(valid, weights, 0.)
        return dict(representation=(history*weights[..., None]).sum(1), attention=weights,
                    scores=torch.where(valid, scores, 0.))
