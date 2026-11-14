"""Independent PEP embedding mechanism, Liu et al., ICLR 2021, Eq. 8/Alg. 1.

Sources: https://arxiv.org/abs/2101.07577 and author repository
ssui-liu/learnable-embed-sizes-for-RecSys, commit
e4d07d2c5e4aaefa95b939b83600adfaf47d7a6f, models/pep_embedding.py.
Host owns vocabulary, initial-weight checkpoint, budget selection, full-model
and optimizer reset, training and evaluation. Dense storage remains allocated;
nonzero parameter counts do not establish memory or latency savings.
"""
import math
from numbers import Real

import torch
from torch import nn


class PrunableEmbedding(nn.Module):
    """Integer IDs [...] -> embeddings [...,D] using learned soft thresholds.

    Search uses sign(V)*ReLU(abs(V)-sigmoid(s)). Threshold sharing is global,
    dimension [1,D], feature [N,1], or feature_dim [N,D]. There is no field
    offset handling. Padding rows always output zero and receive zero gradients.
    threshold_init=-3 is a finite starting point, not a tuned pruning schedule.

    Save initial_weights before search; export_mask() snapshots support. Then
    from_mask(initial_weights, mask) creates a fresh layer for fixed-mask
    retraining with the original weights, without threshold parameters. The
    host must also reset the optimizer and any other model parameters required
    by its experiment. No initial-weight copy is retained inside this layer.
    """
    def __init__(self, initial_weights, threshold_type="global", threshold_init=-3.0,
                 padding_idx=None, *, fixed_mask=None):
        super().__init__()
        values = torch.as_tensor(initial_weights)
        if (values.ndim != 2 or min(values.shape) < 1 or values.is_complex() or
                not bool(torch.isfinite(values).all())):
            raise ValueError("initial_weights must be a finite nonempty real matrix")
        if not values.is_floating_point():
            values = values.float()
        rows, width = values.shape
        shapes = {"global": (1, 1), "dimension": (1, width),
                  "feature": (rows, 1), "feature_dim": (rows, width)}
        if threshold_type not in shapes:
            raise ValueError("invalid threshold_type")
        if isinstance(threshold_init, bool) or not isinstance(threshold_init, Real) or not math.isfinite(threshold_init):
            raise ValueError("threshold_init must be finite and real")
        if padding_idx is not None and (not isinstance(padding_idx, int) or isinstance(padding_idx, bool) or
                                        not 0 <= padding_idx < rows):
            raise ValueError("padding_idx must be a vocabulary row or None")
        self.retraining = fixed_mask is not None
        mask = torch.ones_like(values, dtype=torch.bool) if fixed_mask is None else torch.as_tensor(fixed_mask, device=values.device)
        if mask.dtype != torch.bool or mask.shape != values.shape:
            raise ValueError("fixed_mask must be boolean with the table shape")
        mask = mask.detach().clone()
        if padding_idx is not None:
            mask[padding_idx] = False
        self.register_buffer("mask", mask)
        self.weight = nn.Parameter(torch.where(mask, values.detach(), 0).clone())
        self.threshold_logits = (None if self.retraining else nn.Parameter(
            values.new_full(shapes[threshold_type], float(threshold_init))))

    @classmethod
    def from_mask(cls, initial_weights, mask, padding_idx=None):
        """Start fixed-support retraining from host-supplied initial weights."""
        return cls(initial_weights, padding_idx=padding_idx, fixed_mask=mask)

    def effective_weight(self):
        """Current dense table with exact zeros outside the active support."""
        values = self.weight
        if not self.retraining:
            values = values.sign() * torch.relu(values.abs() - self.threshold_logits.sigmoid())
        return torch.where(self.mask, values, 0)

    def export_mask(self):
        """Detached search support; retraining preserves its fixed mask at zero."""
        return (self.mask if self.retraining else self.effective_weight() != 0).detach().clone()

    def nonzero_count(self):
        """Count current effective nonzeros, not dense allocation or mask size."""
        return torch.count_nonzero(self.effective_weight())

    def forward(self, ids):
        if ids.dtype not in (torch.int32, torch.int64) or bool(((ids < 0) | (ids >= self.weight.shape[0])).any()):
            raise ValueError("ids must be valid integer vocabulary rows")
        return nn.functional.embedding(ids, self.effective_weight())
