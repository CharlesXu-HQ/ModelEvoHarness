"""Original native APG conditional linear and PPNet prior-conditioned gate.

Reference: reczoo/FuxiCTR at 183da1c944ba8930fcdcd54114837c081959d748,
model_zoo/APG/src/APG.py and model_zoo/PEPNet/src/PPNet.py. Independent
hidden-state adapters; no embedding tables, prediction heads or training.
"""
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


def _positive(*values):
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError('dimensions must be positive integers')


def _matrix(x, width):
    if x.ndim != 2 or x.shape[-1] != width:
        raise ValueError(f'expected [B,{width}] tensor')


class AdaptiveLinear(nn.Module):
    """(features[B,I], condition[B,C]) -> [B,O] via generated linear weights.

    Full: x @ W(c). Low-rank: x @ U @ S(c) @ V, with shared U/V.
    Optional overparam_dim factors U/V into products. Generated bias is
    appended after flattened W/S in the affine generator; otherwise bias is
    shared. Self-wise conditioning is explicit: call layer(x,x).
    """
    def __init__(self, input_dim: int, output_dim: int, condition_dim: int,
                 rank: int | None = None, generated_bias: bool = False,
                 overparam_dim: int | None = None):
        super().__init__()
        _positive(input_dim, output_dim, condition_dim)
        if rank is not None:
            _positive(rank)
            if rank > min(input_dim, output_dim):
                raise ValueError('rank cannot exceed input or output dimensions')
        if overparam_dim is not None:
            _positive(overparam_dim)
            if rank is None or overparam_dim < rank:
                raise ValueError('overparam_dim requires a rank and must be at least that rank')
        self.input_dim, self.output_dim, self.condition_dim = input_dim, output_dim, condition_dim
        self.rank, self.generated_bias, self.overparam_dim = rank, generated_bias, overparam_dim
        self.matrix_size = rank * rank if rank is not None else input_dim * output_dim
        self.generator = nn.Linear(condition_dim, self.matrix_size + (output_dim if generated_bias else 0))
        if rank is not None:
            if overparam_dim is None:
                self.input_factor = self._parameter(input_dim, rank)
                self.output_factor = self._parameter(rank, output_dim)
            else:
                self.input_left = self._parameter(input_dim, overparam_dim)
                self.input_right = self._parameter(overparam_dim, rank)
                self.output_left = self._parameter(rank, overparam_dim)
                self.output_right = self._parameter(overparam_dim, output_dim)
        self.bias = None if generated_bias else nn.Parameter(torch.zeros(output_dim))

    @staticmethod
    def _parameter(rows, cols):
        value = nn.Parameter(torch.empty(rows, cols))
        nn.init.xavier_uniform_(value)
        return value

    def forward(self, features, condition):
        _matrix(features, self.input_dim)
        _matrix(condition, self.condition_dim)
        if features.shape[0] != condition.shape[0]:
            raise ValueError('features and condition must have the same batch size')
        generated = self.generator(condition)
        bias = generated[:, self.matrix_size:] if self.generated_bias else self.bias
        if self.rank is None:
            matrix = generated[:, :self.matrix_size].reshape(features.shape[0], self.input_dim, self.output_dim)
            output = torch.bmm(features[:, None], matrix).squeeze(1)
        else:
            u = self.input_factor if self.overparam_dim is None else self.input_left @ self.input_right
            v = self.output_factor if self.overparam_dim is None else self.output_left @ self.output_right
            center = generated[:, :self.matrix_size].reshape(features.shape[0], self.rank, self.rank)
            output = torch.bmm((features @ u)[:, None], center).squeeze(1) @ v
        return output + bias


class PriorConditionedGate(nn.Module):
    """(hidden[B,H], features[B,F], prior[B,P]) -> [B,H].

    hidden * 2*sigmoid(MLP([stopgrad(features), prior])). detach_features
    defaults True; prior and hidden remain differentiable. Instantiate a gate
    per hidden depth, reusing the host's base features and declared priors.
    """
    def __init__(self, feature_dim: int, prior_dim: int, hidden_dim: int,
                 gate_hidden_dim: int, detach_features: bool = True):
        super().__init__()
        _positive(feature_dim, prior_dim, hidden_dim, gate_hidden_dim)
        self.feature_dim, self.prior_dim, self.hidden_dim = feature_dim, prior_dim, hidden_dim
        self.detach_features = detach_features
        self.gate_in = nn.Linear(feature_dim + prior_dim, gate_hidden_dim)
        self.gate_out = nn.Linear(gate_hidden_dim, hidden_dim)

    def forward(self, hidden, features, prior):
        _matrix(hidden, self.hidden_dim)
        _matrix(features, self.feature_dim)
        _matrix(prior, self.prior_dim)
        if hidden.shape[0] != features.shape[0] or hidden.shape[0] != prior.shape[0]:
            raise ValueError('hidden, features and prior must have the same batch size')
        base = features.detach() if self.detach_features else features
        gate = self.gate_out(F.relu(self.gate_in(torch.cat((base, prior), -1))))
        return hidden * (2 * torch.sigmoid(gate))
