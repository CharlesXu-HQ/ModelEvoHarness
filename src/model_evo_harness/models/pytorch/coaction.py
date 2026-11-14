"""Independent native CAN feature-pair micro-MLP, without host embedding tables.

Reference: CAN (arXiv:2011.05625), section 4; author implementation
CAN-Paper/Co-Action-Network at 98a7219105568b9420cf28467877d08a1c56ae69.
The paper sums orders; the author's gen_coaction concatenates them. Both
readout choices are explicit here. No third-party model code is imported.
"""
from numbers import Integral

import torch
from torch import nn


class CoAction(nn.Module):
    """(parameters[B,P], feed[B,T,D], valid[B,T]) -> [B,T,C].

    The host supplies a separate parameter vector for each sample. For layer
    l with widths (d[l-1], d[l]), its row-major matrix W_l[d[l-1],d[l]] is
    followed immediately by b_l[d[l]] when use_bias=True; then comes layer
    l+1. Thus P=sum(d[l-1]*d[l] + (d[l] if use_bias else 0)).

    Each order q=1..order starts from feed**q and reuses the same matrices
    and biases. Layers compute h@W+b, with tanh on every nonfinal layer.
    Concatenate all layer outputs within each order, then concatenate orders
    in ascending order (default), or sum their readouts. C is sum(layer_dims)
    times order for concat, and sum(layer_dims) for sum.

    True in valid denotes real input. Padding is zeroed before powers and
    after readout; all-invalid samples also ignore their parameter vector.
    Inputs must have matching floating dtypes; valid must be boolean. The
    component owns no learned weights and performs no history reduction.
    """
    def __init__(self, input_dim: int, layer_dims, order: int = 1,
                 order_reduction: str = 'concat', use_bias: bool = False):
        super().__init__()
        try:
            layer_dims = tuple(layer_dims)
        except TypeError as exc:
            raise ValueError('layer_dims must contain positive integer dimensions') from exc
        if not layer_dims or any(not isinstance(v, Integral) or isinstance(v, bool) or v < 1
                                 for v in (input_dim, order, *layer_dims)):
            raise ValueError('input_dim, order and layer_dims must be positive integers')
        if order_reduction not in ('concat', 'sum'):
            raise ValueError('order_reduction must be concat or sum')
        if not isinstance(use_bias, bool):
            raise ValueError('use_bias must be boolean')
        self.input_dim, self.layer_dims = input_dim, layer_dims
        self.order, self.order_reduction, self.use_bias = order, order_reduction, use_bias
        widths = (input_dim, *layer_dims)
        self.parameter_dim = sum(a*b + (b if use_bias else 0) for a, b in zip(widths, widths[1:]))
        self.output_dim = sum(layer_dims) * (order if order_reduction == 'concat' else 1)

    def forward(self, parameters, feed, valid):
        if not all(isinstance(x, torch.Tensor) for x in (parameters, feed, valid)):
            raise TypeError('parameters, feed and valid must be tensors')
        if not parameters.is_floating_point() or not feed.is_floating_point() or parameters.dtype != feed.dtype:
            raise TypeError('parameters and feed must have the same floating dtype')
        if valid.dtype != torch.bool:
            raise TypeError('valid must be boolean')
        if parameters.device != feed.device or valid.device != feed.device:
            raise ValueError('parameters, feed and valid must be on the same device')
        if parameters.ndim != 2 or parameters.shape[1] != self.parameter_dim:
            raise ValueError(f'parameters must have shape [B,{self.parameter_dim}]')
        if feed.ndim != 3 or feed.shape[2] != self.input_dim:
            raise ValueError(f'feed must have shape [B,T,{self.input_dim}]')
        if parameters.shape[0] != feed.shape[0]:
            raise ValueError('parameters and feed must have the same batch size')
        if valid.ndim != 2 or valid.shape != feed.shape[:2]:
            raise ValueError('valid must have shape [B,T] matching feed')
        feed = torch.where(valid[..., None], feed, torch.zeros_like(feed))
        parameters = torch.where(valid.any(-1, keepdim=True), parameters, torch.zeros_like(parameters))
        layers, offset, width = [], 0, self.input_dim
        for size in self.layer_dims:
            count = width * size
            matrix = parameters[:, offset:offset+count].reshape(parameters.shape[0], width, size)
            offset += count
            bias = parameters[:, offset:offset+size] if self.use_bias else None
            offset += size if self.use_bias else 0
            layers.append((matrix, bias))
            width = size
        orders = []
        for power in range(1, self.order + 1):
            hidden, outputs = feed.pow(power), []
            for index, (matrix, bias) in enumerate(layers):
                hidden = torch.bmm(hidden, matrix)
                if bias is not None:
                    hidden = hidden + bias[:, None]
                if index < len(layers) - 1:
                    hidden = torch.tanh(hidden)
                outputs.append(hidden)
            orders.append(torch.cat(outputs, dim=-1))
        output = torch.cat(orders, dim=-1) if self.order_reduction == 'concat' else torch.stack(orders).sum(0)
        return torch.where(valid[..., None], output, torch.zeros_like(output))
