"""Native composable cross-mixture and input-aware FM blocks.

Independent implementations of DCN-Mix, IFM and DIFM mechanisms. Hosts own
feature embeddings, task heads, global bias, loss and optimizers; no external
model framework is imported. Inputs must be finite floating tensors.
"""

from numbers import Integral

import torch
from torch import nn


def _positive(*values):
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError('dimensions must be positive integers')


def _shape(x, dimensions):
    if not x.is_floating_point() or x.ndim != len(dimensions) + 1 or tuple(x.shape[1:]) != dimensions:
        raise ValueError(f'expected floating [B,{dimensions}] tensor')
    if not torch.isfinite(x).all():
        raise ValueError('inputs must be finite; sanitize padding before this block')
    return x


class CrossNetMix(nn.Module):
    """Dense [B,D] -> [B,D] nonlinear low-rank cross experts.

    At layer l, expert e is U[l,e] tanh(C[l,e] tanh(V[l,e]^T h)).
    Softmax(h @ router) mixes experts; update is h + x0*(mixture+b[l]).
    Expert parameters differ per layer; the bias-free router is shared across
    layers. return_routing=True also returns probabilities [B,L,E]. This is
    a cross block, with no parallel deep branch or classifier. Rank/expert
    choices alter both capacity and cost and need controlled comparison.
    """
    def __init__(self, input_dim, rank=8, num_experts=4, num_layers=2):
        super().__init__()
        _positive(input_dim, rank, num_experts, num_layers)
        self.input_dim, self.num_layers = input_dim, num_layers
        self.u = nn.Parameter(torch.empty(num_layers, num_experts, input_dim, rank))
        self.v = nn.Parameter(torch.empty(num_layers, num_experts, input_dim, rank))
        self.c = nn.Parameter(torch.empty(num_layers, num_experts, rank, rank))
        self.router = nn.Parameter(torch.empty(input_dim, num_experts))
        self.bias = nn.Parameter(torch.zeros(num_layers, input_dim))
        with torch.no_grad():
            for bank in (self.u, self.v, self.c):
                for experts in bank:
                    for matrix in experts:
                        nn.init.xavier_uniform_(matrix)
        nn.init.xavier_uniform_(self.router)

    def forward(self, x, return_routing=False):
        x = _shape(x, (self.input_dim,))
        hidden, routing = x, []
        for index in range(self.num_layers):
            weights = torch.softmax(hidden @ self.router, dim=-1)
            low = torch.tanh(torch.einsum('bd,edr->ber', hidden, self.v[index]))
            low = torch.tanh(torch.einsum('ers,bes->ber', self.c[index], low))
            experts = torch.einsum('edr,ber->bed', self.u[index], low)
            hidden = hidden + x * ((experts * weights[..., None]).sum(1) + self.bias[index])
            routing.append(weights)
        return (hidden, torch.stack(routing, 1)) if return_routing else hidden


class InputAwareFM(nn.Module):
    """Host embeddings [B,F,D] and first-order field terms [B,F] -> logits [B].

    One factor a[B,F] jointly weights first-order terms and embeddings:
    sum_i(a_i*w_i) + sum_{i<j}<a_i*e_i,a_j*e_j>. No global bias is added.
    IFM: a=F*softmax(MLP(flat(e)) @ bit_output). DIFM: an MLP factor plus
    a multi-head field-attention factor, without final softmax or positivity.
    Attention uses Q/K/V, scaled dot products, a projected residual and ReLU;
    field_factors/factor_components expose factors for controlled ablations.
    All F slots must be observed; this block does not interpret missing fields.
    """
    def __init__(self, num_fields, embedding_dim, hidden_dims=(32,), mode='ifm', num_heads=2):
        super().__init__()
        _positive(num_fields, embedding_dim, *hidden_dims)
        if num_fields < 2 or mode not in ('ifm', 'difm'):
            raise ValueError('at least two fields and ifm/difm mode required')
        if mode == 'difm':
            _positive(num_heads)
            if embedding_dim % num_heads:
                raise ValueError('embedding_dim must be divisible by num_heads')
        self.num_fields, self.embedding_dim, self.mode = num_fields, embedding_dim, mode
        self.num_heads = num_heads
        width = num_fields * embedding_dim
        layers = []
        for size in hidden_dims:
            layers.extend((nn.Linear(width, size), nn.ReLU()))
            width = size
        self.bit_mlp = nn.Sequential(*layers)
        self.bit_output = nn.Parameter(torch.empty(width, num_fields))
        nn.init.xavier_uniform_(self.bit_output)
        if mode == 'difm':
            for name in ('query', 'key', 'value', 'residual'):
                parameter = nn.Parameter(torch.empty(embedding_dim, embedding_dim))
                nn.init.xavier_uniform_(parameter)
                setattr(self, name, parameter)
            self.vector_output = nn.Parameter(torch.empty(num_fields * embedding_dim, num_fields))
            nn.init.xavier_uniform_(self.vector_output)

    def factor_components(self, tokens):
        """Return unnormalized (bit, vector) [B,F] factors; vector=0 for IFM."""
        tokens = _shape(tokens, (self.num_fields, self.embedding_dim))
        bit = self.bit_mlp(tokens.flatten(1)) @ self.bit_output
        if self.mode == 'ifm':
            return bit, torch.zeros_like(bit)
        batch, fields, dim = tokens.shape
        width = dim // self.num_heads
        q, k, v = [(tokens @ matrix).reshape(batch, fields, self.num_heads, width).transpose(1, 2)
                   for matrix in (self.query, self.key, self.value)]
        weights = torch.softmax((q @ k.transpose(-1, -2)) / width ** .5, dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, fields, dim)
        attended = torch.relu(attended + tokens @ self.residual)
        return bit, attended.flatten(1) @ self.vector_output

    def field_factors(self, tokens):
        bit, vector = self.factor_components(tokens)
        return self.num_fields * torch.softmax(bit, -1) if self.mode == 'ifm' else bit + vector

    def forward(self, tokens, first_order_fields):
        first_order_fields = _shape(first_order_fields, (self.num_fields,))
        if tokens.shape[0] != first_order_fields.shape[0]:
            raise ValueError('tokens and first_order_fields batch sizes must match')
        factors = self.field_factors(tokens)
        weighted = tokens * factors[..., None]
        pair_sum = .5 * (weighted.sum(1).square() - weighted.square().sum(1)).sum(-1)
        return (first_order_fields * factors).sum(-1) + pair_sum


class DualInputAwareFM(InputAwareFM):
    """DIFM entry point; same input/output contract as InputAwareFM."""
    def __init__(self, num_fields, embedding_dim, hidden_dims=(32,), num_heads=2):
        super().__init__(num_fields, embedding_dim, hidden_dims, mode='difm', num_heads=num_heads)
