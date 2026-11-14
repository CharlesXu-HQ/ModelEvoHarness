"""Native representation recipes; host owns vocabulary identity and feature timing."""
import torch
from torch import nn


class PretrainedIDFusion(nn.Module):
    """Aligned pretrained table with init/sum/concat ID fusion and a fixed padding ID.

    All rows, including OOV/pretrained-only rows, retain the same availability when
    freeze_pretrained changes. The host supplies a single versioned ID mapping.
    """
    def __init__(self, pretrained_vectors, output_dim, mode="init", freeze_pretrained=True, padding_idx=0):
        super().__init__()
        values = torch.as_tensor(pretrained_vectors)
        if (values.ndim != 2 or min(values.shape) < 1 or values.is_complex() or
                not bool(torch.isfinite(values).all())):
            raise ValueError("pretrained_vectors must be a finite nonempty real matrix")
        if not values.is_floating_point():
            values = values.float()
        if not isinstance(output_dim, int) or isinstance(output_dim, bool) or output_dim < 1:
            raise ValueError("output_dim must be positive")
        if mode not in ("init", "sum", "concat") or not isinstance(freeze_pretrained, bool):
            raise ValueError("invalid mode or freeze_pretrained")
        if padding_idx is not None and (not isinstance(padding_idx, int) or isinstance(padding_idx, bool) or
                                        not 0 <= padding_idx < values.shape[0]):
            raise ValueError("padding_idx must be a valid row or None")
        self.mode, self.padding_idx = mode, padding_idx
        self.pretrained = nn.Embedding.from_pretrained(values.detach().clone(), freeze=freeze_pretrained,
                                                       padding_idx=padding_idx)
        self.id_embedding = (nn.Embedding(values.shape[0], output_dim, padding_idx=padding_idx,
                                          dtype=values.dtype, device=values.device) if mode != "init" else None)
        projection_dim = values.shape[1] + (output_dim if mode == "concat" else 0)
        self.projection = (nn.Linear(projection_dim, output_dim, bias=False, dtype=values.dtype,
                                      device=values.device) if projection_dim != output_dim or mode == "concat"
                           else nn.Identity())

    def forward(self, ids):
        if ids.dtype not in (torch.int32, torch.int64) or bool(((ids < 0) | (ids >= self.pretrained.num_embeddings)).any()):
            raise ValueError("ids must be valid integer vocabulary rows")
        values = self.pretrained(ids)
        if self.mode == "concat":
            result = self.projection(torch.cat([values, self.id_embedding(ids)], -1))
        else:
            result = self.projection(values)
            if self.mode == "sum":
                result = result + self.id_embedding(ids)
        return (torch.where((ids != self.padding_idx)[..., None], result, torch.zeros_like(result))
                if self.padding_idx is not None else result)


class ResponseMLP(nn.Module):
    """Dense features [B,D] to raw binary logit [B]; empty hidden_dims gives LR."""
    def __init__(self, input_dim, hidden_dims=()):
        super().__init__()
        widths = [input_dim, *hidden_dims]
        if any(not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in widths):
            raise ValueError("dimensions must be positive integers")
        self.input_dim = input_dim
        layers = []
        for left, right in zip(widths, widths[1:]):
            layers += [nn.Linear(left, right), nn.ReLU()]
        self.network = nn.Sequential(*layers, nn.Linear(widths[-1], 1))

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != self.input_dim or not bool(torch.isfinite(features).all()):
            raise ValueError("features must be finite [B,input_dim]")
        return self.network(features).squeeze(-1)
